#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// check if a buffer is a CUDA device buffer (not split, not host)
GGML_BACKEND_API bool ggml_backend_buffer_is_cuda_device(ggml_backend_buffer_t buffer);

// get the CUDA device ordinal for a CUDA device buffer (-1 if not a CUDA buffer)
GGML_BACKEND_API int ggml_backend_cuda_buffer_get_device(ggml_backend_buffer_t buffer);

// GPU Direct Storage (GDS) -- direct NVMe-to-GPU DMA transfers (Linux + NVIDIA only)
#ifdef GGML_CUDA_USE_GDS

#include <sys/types.h>

// opaque GDS file handle
typedef void * ggml_cuda_gds_file_handle_t;

// minimum tensor size to use GDS (avoids setup overhead for small tensors)
#define GGML_CUDA_GDS_MIN_TRANSFER_SIZE 65536

GGML_BACKEND_API bool                       ggml_cuda_gds_init(void);
GGML_BACKEND_API void                       ggml_cuda_gds_shutdown(void);
GGML_BACKEND_API bool                       ggml_cuda_gds_available(void);
GGML_BACKEND_API ggml_cuda_gds_file_handle_t ggml_cuda_gds_register_file(int fd);
GGML_BACKEND_API void                       ggml_cuda_gds_deregister_file(ggml_cuda_gds_file_handle_t handle);
GGML_BACKEND_API int                        ggml_cuda_gds_register_buffer(void * dev_ptr, size_t size);
GGML_BACKEND_API void                       ggml_cuda_gds_deregister_buffer(void * dev_ptr);
GGML_BACKEND_API ssize_t                    ggml_cuda_gds_read(ggml_cuda_gds_file_handle_t handle, void * dev_ptr, size_t size, off_t file_offset, int device);

#endif // GGML_CUDA_USE_GDS

#ifdef  __cplusplus
}
#endif
