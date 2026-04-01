// NVIDIA GPU Direct Storage (GDS) implementation for Ollama.
// Dynamically loads libcufile.so at runtime via dlopen -- no link-time dependency.
// Falls back gracefully when GDS is unavailable (no library, no driver, non-NVMe, etc).

#ifdef GGML_CUDA_USE_GDS

#include "gds.cuh"
#include "common.cuh"

#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <atomic>
#include <chrono>

// cuFile type definitions -- we define these locally to avoid requiring cufile.h at build time
// on systems that don't have the GDS SDK installed. The ABI is stable.

typedef int CUfileError_code_t;

struct CUfileError_t {
    CUfileError_code_t err;
    CUresult           cu_err;
};

enum CUfileFileHandleType_t {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
};

struct CUfileDescr_t {
    CUfileFileHandleType_t type;
    union {
        int          fd;
        void *       handle;
    } handle;
    // Padding to match cuFile ABI
    unsigned int reserved[16];
};

typedef void * CUfileHandle_t;

#define CU_FILE_SUCCESS 0

// Function pointer types for the cuFile API
typedef CUfileError_t  (*fn_cuFileDriverOpen_t)(void);
typedef CUfileError_t  (*fn_cuFileDriverClose_t)(void);
typedef CUfileError_t  (*fn_cuFileHandleRegister_t)(CUfileHandle_t *, CUfileDescr_t *);
typedef void           (*fn_cuFileHandleDeregister_t)(CUfileHandle_t);
typedef CUfileError_t  (*fn_cuFileBufRegister_t)(const void *, size_t, int);
typedef CUfileError_t  (*fn_cuFileBufDeregister_t)(const void *);
typedef ssize_t        (*fn_cuFileRead_t)(CUfileHandle_t, void *, size_t, off_t, off_t);

// Dynamically loaded function pointers
static struct gds_api_t {
    void * lib_handle = nullptr;

    fn_cuFileDriverOpen_t        driverOpen        = nullptr;
    fn_cuFileDriverClose_t       driverClose       = nullptr;
    fn_cuFileHandleRegister_t    handleRegister     = nullptr;
    fn_cuFileHandleDeregister_t  handleDeregister   = nullptr;
    fn_cuFileBufRegister_t       bufRegister        = nullptr;
    fn_cuFileBufDeregister_t     bufDeregister      = nullptr;
    fn_cuFileRead_t              fileRead           = nullptr;
} gds_api;

static std::once_flag gds_init_flag;
static std::atomic<bool> gds_initialized{false};
static std::atomic<bool> gds_is_available{false};
static bool gds_stats_enabled = false;

static inline size_t align_down(size_t val, size_t alignment) {
    return val & ~(alignment - 1);
}

static inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static void gds_do_init() {
    // Check environment variable to disable GDS
    const char * no_gds = std::getenv("GGML_CUDA_NO_GDS");
    if (no_gds && *no_gds) {
        GGML_LOG_INFO("%s: GDS disabled via GGML_CUDA_NO_GDS\n", __func__);
        return;
    }

    gds_stats_enabled = (std::getenv("GGML_CUDA_GDS_STATS") != nullptr);

    // Try to load libcufile.so
    gds_api.lib_handle = dlopen("libcufile.so.0", RTLD_LAZY);
    if (!gds_api.lib_handle) {
        gds_api.lib_handle = dlopen("libcufile.so", RTLD_LAZY);
    }
    if (!gds_api.lib_handle) {
        GGML_LOG_DEBUG("%s: libcufile.so not found, GDS unavailable\n", __func__);
        return;
    }

    // Resolve cuFile API symbols
    gds_api.driverOpen       = (fn_cuFileDriverOpen_t)      dlsym(gds_api.lib_handle, "cuFileDriverOpen");
    gds_api.driverClose      = (fn_cuFileDriverClose_t)     dlsym(gds_api.lib_handle, "cuFileDriverClose");
    gds_api.handleRegister   = (fn_cuFileHandleRegister_t)  dlsym(gds_api.lib_handle, "cuFileHandleRegister");
    gds_api.handleDeregister = (fn_cuFileHandleDeregister_t)dlsym(gds_api.lib_handle, "cuFileHandleDeregister");
    gds_api.bufRegister      = (fn_cuFileBufRegister_t)     dlsym(gds_api.lib_handle, "cuFileBufRegister");
    gds_api.bufDeregister    = (fn_cuFileBufDeregister_t)   dlsym(gds_api.lib_handle, "cuFileBufDeregister");
    gds_api.fileRead         = (fn_cuFileRead_t)            dlsym(gds_api.lib_handle, "cuFileRead");

    if (!gds_api.driverOpen || !gds_api.driverClose ||
        !gds_api.handleRegister || !gds_api.handleDeregister ||
        !gds_api.bufRegister || !gds_api.bufDeregister ||
        !gds_api.fileRead) {
        GGML_LOG_DEBUG("%s: failed to resolve all cuFile symbols\n", __func__);
        dlclose(gds_api.lib_handle);
        gds_api.lib_handle = nullptr;
        return;
    }

    // Initialize the GDS driver
    CUfileError_t err = gds_api.driverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        GGML_LOG_DEBUG("%s: cuFileDriverOpen failed (err=%d), GDS unavailable "
                       "(nvidia-fs kernel module may not be loaded)\n",
                       __func__, err.err);
        dlclose(gds_api.lib_handle);
        gds_api.lib_handle = nullptr;
        return;
    }

    gds_is_available.store(true, std::memory_order_release);
    GGML_LOG_INFO("%s: GPU Direct Storage initialized successfully\n", __func__);
}

bool ggml_cuda_gds_init(void) {
    std::call_once(gds_init_flag, gds_do_init);
    gds_initialized.store(true, std::memory_order_release);
    return gds_is_available.load(std::memory_order_acquire);
}

void ggml_cuda_gds_shutdown(void) {
    if (!gds_initialized.load(std::memory_order_acquire)) {
        return;
    }
    if (gds_is_available.load(std::memory_order_acquire) && gds_api.driverClose) {
        gds_api.driverClose();
    }
    if (gds_api.lib_handle) {
        dlclose(gds_api.lib_handle);
        gds_api.lib_handle = nullptr;
    }
    gds_is_available.store(false, std::memory_order_release);
}

bool ggml_cuda_gds_available(void) {
    return gds_is_available.load(std::memory_order_acquire);
}

ggml_cuda_gds_file_handle_t ggml_cuda_gds_register_file(int fd) {
    if (!ggml_cuda_gds_available()) {
        return nullptr;
    }

    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;

    CUfileHandle_t handle = nullptr;
    CUfileError_t err = gds_api.handleRegister(&handle, &descr);
    if (err.err != CU_FILE_SUCCESS) {
        GGML_LOG_DEBUG("%s: cuFileHandleRegister failed for fd=%d (err=%d)\n",
                       __func__, fd, err.err);
        return nullptr;
    }

    return static_cast<ggml_cuda_gds_file_handle_t>(handle);
}

void ggml_cuda_gds_deregister_file(ggml_cuda_gds_file_handle_t handle) {
    if (handle && gds_api.handleDeregister) {
        gds_api.handleDeregister(static_cast<CUfileHandle_t>(handle));
    }
}

int ggml_cuda_gds_register_buffer(void * dev_ptr, size_t size) {
    if (!ggml_cuda_gds_available() || !dev_ptr || size == 0) {
        return -1;
    }

    CUfileError_t err = gds_api.bufRegister(dev_ptr, size, 0);
    if (err.err != CU_FILE_SUCCESS) {
        GGML_LOG_DEBUG("%s: cuFileBufRegister failed (ptr=%p, size=%zu, err=%d)\n",
                       __func__, dev_ptr, size, err.err);
        return -1;
    }

    return 0;
}

void ggml_cuda_gds_deregister_buffer(void * dev_ptr) {
    if (dev_ptr && gds_api.bufDeregister) {
        gds_api.bufDeregister(dev_ptr);
    }
}

ssize_t ggml_cuda_gds_read(
        ggml_cuda_gds_file_handle_t handle,
        void * dev_ptr,
        size_t size,
        off_t file_offset,
        int device) {
    if (!handle || !dev_ptr || size == 0) {
        return -1;
    }

    ggml_cuda_set_device(device);

    auto t_start = std::chrono::high_resolution_clock::now();

    const size_t aligned_offset = align_down((size_t)file_offset, GGML_CUDA_GDS_ALIGNMENT);
    const size_t prefix_bytes   = (size_t)file_offset - aligned_offset;
    const bool   offset_aligned = (prefix_bytes == 0);
    const bool   ptr_aligned    = (((uintptr_t)dev_ptr) % GGML_CUDA_GDS_ALIGNMENT) == 0;
    const bool   size_aligned   = (size % GGML_CUDA_GDS_ALIGNMENT) == 0;

    ssize_t result;

    if (offset_aligned && ptr_aligned && size_aligned) {
        // Fast path: everything is aligned, read directly into destination
        result = gds_api.fileRead(
            static_cast<CUfileHandle_t>(handle),
            dev_ptr,
            size,
            (off_t)aligned_offset,
            0  // dev_ptr offset
        );
    } else {
        // Bounce path: use an aligned temporary GPU buffer
        const size_t total_read = align_up(prefix_bytes + size, GGML_CUDA_GDS_ALIGNMENT);

        void * bounce_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&bounce_buf, total_read));

        // Register bounce buffer for GDS
        CUfileError_t reg_err = gds_api.bufRegister(bounce_buf, total_read, 0);
        if (reg_err.err != CU_FILE_SUCCESS) {
            GGML_LOG_DEBUG("%s: bounce buffer registration failed (err=%d)\n",
                           __func__, reg_err.err);
            cudaFree(bounce_buf);
            return -1;
        }

        ssize_t bytes_read = gds_api.fileRead(
            static_cast<CUfileHandle_t>(handle),
            bounce_buf,
            total_read,
            (off_t)aligned_offset,
            0  // dev_ptr offset
        );

        if (bytes_read >= (ssize_t)(prefix_bytes + size)) {
            // Copy the relevant portion from bounce buffer to destination
            CUDA_CHECK(cudaMemcpy(
                dev_ptr,
                (char *)bounce_buf + prefix_bytes,
                size,
                cudaMemcpyDeviceToDevice
            ));
            result = (ssize_t)size;
        } else {
            GGML_LOG_DEBUG("%s: cuFileRead returned %zd, expected >= %zu\n",
                           __func__, bytes_read, prefix_bytes + size);
            result = -1;
        }

        gds_api.bufDeregister(bounce_buf);
        cudaFree(bounce_buf);
    }

    if (gds_stats_enabled && result > 0) {
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double gbps = ((double)size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
        GGML_LOG_INFO("%s: GDS read %zu bytes in %.2f ms (%.2f GB/s)%s\n",
                      __func__, size, ms, gbps,
                      (offset_aligned && ptr_aligned && size_aligned) ? "" : " [bounce]");
    }

    return result;
}

#endif // GGML_CUDA_USE_GDS
