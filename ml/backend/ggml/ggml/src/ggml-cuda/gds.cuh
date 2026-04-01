#pragma once

// Internal GDS constants used by gds.cu.
// Public API declarations are in ggml-cuda.h.

#ifdef GGML_CUDA_USE_GDS

#include <cstddef>

// 4KB alignment required by GDS for file offsets, buffer addresses, and transfer sizes
static constexpr size_t GGML_CUDA_GDS_ALIGNMENT = 4096;

#endif // GGML_CUDA_USE_GDS
