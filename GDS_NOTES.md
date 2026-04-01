# NVIDIA GPU Direct Storage (GDS) for Ollama

Internal working notes for the GDS feature branch. To be removed before upstream PR.

## What This Does

Adds a third model-loading path to Ollama that uses NVIDIA's cuFile API to DMA model
weights directly from NVMe storage into GPU VRAM, bypassing CPU and system memory:

```
Traditional:  NVMe -> Page Cache -> CPU Memory -> cudaMemcpyAsync -> GPU VRAM
GDS:          NVMe -> PCIe DMA -> GPU VRAM  (single hop, no CPU involvement)
```

## Why It Matters

### Serverless GPU / Scale-to-Zero

This is the killer use case. Platforms like RunPod Serverless, Modal, Baseten, and Replicate
scale GPU containers to zero when idle and cold-start on incoming requests. The cold start
sequence is:

1. Pull container image (usually cached on node)
2. **Load model weights from disk into GPU** -- the bottleneck
3. First token generation

Step 2 dominates cold start latency. GDS directly reduces it by eliminating the CPU memory
bounce. For bursty workloads where cold starts are frequent, this means:

- Lower time-to-first-token for every scale-up event
- Less wasted GPU billing (GPU is allocated but idle during model loading)
- Faster autoscaling response to traffic spikes

### Server Deployments Generally

Ollama is heavily used in production Linux GPU deployments: Docker containers, Kubernetes
clusters, cloud GPU instances (AWS, GCP, Lambda Labs). These are all Linux environments
where GDS is available. The feature is purely additive -- zero impact on non-Linux users.

## Test Hardware

- **GPU**: NVIDIA GeForce RTX 5090 (Blackwell, PCIe 5.0 x16)
- **NVMe**: Samsung PM9E1 2TB (PCIe 5.0, OEM variant of 990 EVO Plus)
- **System**: Currently Windows 11 + WSL2 (cannot test GDS here)
- **Target**: Ubuntu 24.04 LTS dual-boot for testing

This hardware is ideal -- PCIe 5.0 on both GPU and NVMe gives theoretical ~14 GB/s
sequential read from drive to GPU.

## Expected Performance

| Scenario | Estimated Improvement |
|---|---|
| Cold load, 70B Q4 (~37GB) | 20-40% faster (5-8s -> 3-5s) |
| Cold load, 8B Q4 (~4.3GB) | 10-25% faster (~1s -> ~0.5s) |
| NVMe RAID (2-4 drives) | Up to 50%+ (CPU was the bottleneck) |
| Warm load (model in page cache) | Minimal or no improvement |
| Small models (<1GB) | Negligible (setup overhead dominates) |

These are estimates. Real benchmarks required before upstream PR.

## TODO: Pre-PR Checklist

### Setup
- [ ] Install Ubuntu 24.04 LTS (dual-boot alongside Windows)
- [ ] Install NVIDIA driver + CUDA toolkit + GDS packages:
  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update
  sudo apt install cuda nvidia-gds
  sudo reboot
  ```
- [ ] Verify: `nvidia-smi` shows RTX 5090
- [ ] Verify: `ls /dev/nvidia-fs*` shows GDS device nodes
- [ ] Clone this branch: `git clone -b feat/nvidia-gds https://github.com/maxwbuckley/ollama.git`

### Build
- [ ] Build Ollama with GDS enabled:
  ```bash
  cmake -B build -DGGML_CUDA_GDS=ON
  cmake --build build
  ```
- [ ] Fix any compilation issues (cuFile ABI struct layout, include paths, etc.)
- [ ] Verify GDS initializes: `OLLAMA_GPU_DIRECT_STORAGE=1 OLLAMA_DEBUG=1 ./ollama run llama3:8b "hello"`
  - Should see: `GPU Direct Storage initialized successfully`
  - Should see: `using GPU Direct Storage for model loading`

### Correctness Verification
- [ ] Load model WITHOUT GDS, capture output checksums:
  ```bash
  GGML_CUDA_NO_GDS=1 ollama run llama3:8b "The quick brown fox" --verbose
  ```
- [ ] Load model WITH GDS, capture output checksums:
  ```bash
  OLLAMA_GPU_DIRECT_STORAGE=1 ollama run llama3:8b "The quick brown fox" --verbose
  ```
- [ ] Verify bit-exact identical output (same tokens, same logits)
- [ ] Test with split model files (should gracefully skip GDS, fall back)
- [ ] Test on non-NVMe filesystem (e.g., USB drive) -- should fall back gracefully

### Benchmarking (following repo CLAUDE.md benchmarking rules)
- [ ] Benchmark script that:
  - Drops page cache between runs: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
  - Measures time from load start to first token
  - Runs n>=30 iterations with warmup
  - Reports mean, median, stdev, and Welch's t-test
- [ ] Test matrix:
  - Models: 1B param, 8B param, 70B param (if VRAM allows)
  - Conditions: GDS on vs GDS off (GGML_CUDA_NO_GDS=1)
  - All cold-cache (drop page cache between every run)
- [ ] Capture per-tensor stats: `GGML_CUDA_GDS_STATS=1`
- [ ] Report typical-case and worst-case, not best-case
- [ ] Check variance/stdev between baseline and GDS
- [ ] Lock GPU clocks for stable measurements: `sudo nvidia-smi -lgc <max_clock>`

### PR Preparation
- [ ] Remove this file (GDS_NOTES.md) before PR
- [ ] Write PR description with:
  - Real benchmark numbers (median improvement across full test matrix)
  - The range (e.g., "3-7% typical, up to 26% for minimal inputs")
  - Test hardware specs
  - How to reproduce
- [ ] Ensure CI passes (GDS code is behind `GGML_CUDA_GDS` flag, default OFF)
- [ ] Consider: should we submit to llama.cpp upstream first? The core changes
  to `load_all_data()` and `ggml-cuda` are in their code, not Ollama-specific

## Implementation Architecture

### Files Created
- `ml/backend/ggml/ggml/src/ggml-cuda/gds.cuh` -- Internal alignment constant
- `ml/backend/ggml/ggml/src/ggml-cuda/gds.cu` -- Full GDS implementation

### Files Modified
- `CMakeLists.txt` -- `set(GGML_CUDA_GDS OFF)` default
- `ml/backend/ggml/ggml/src/ggml-cuda/CMakeLists.txt` -- Build option + dlopen linkage
- `ml/backend/ggml/ggml/include/ggml-cuda.h` -- Public API declarations
- `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` -- Buffer inspection wrappers
- `llama/llama.cpp/src/llama-model-loader.cpp` -- GDS path in `load_all_data()`
- `envconfig/config.go` -- `OLLAMA_GPU_DIRECT_STORAGE` env var
- `llm/server.go` -- Force `use_mmap=false` when GDS requested

### Fallback Chain (5 layers)
1. Compile-time: `GGML_CUDA_GDS=OFF` (default) -> no GDS code
2. Runtime - no library: `dlopen("libcufile.so")` fails -> fallback
3. Runtime - no driver: `cuFileDriverOpen()` fails -> fallback
4. Runtime - file registration fails: non-NVMe filesystem -> fallback
5. Per-tensor: `cuFileRead` error -> falls through to staged upload path

### Key Design Decisions
- **dlopen not link-time**: Binary works everywhere, GDS activates only when present
- **Per-tensor fallback**: One bad tensor doesn't break the whole load
- **64KB minimum**: Skip GDS overhead for small tensors (biases, embeddings)
- **Bounce buffer for alignment**: GGUF uses 32-byte alignment, GDS needs 4KB
- **Split buffers skipped in v1**: Multi-GPU tensor distribution deferred to v2

## Environment Variables

| Variable | Purpose |
|---|---|
| `OLLAMA_GPU_DIRECT_STORAGE=1` | Enable GDS (forces mmap off on Linux+CUDA) |
| `GGML_CUDA_NO_GDS=1` | Disable GDS even when compiled in |
| `GGML_CUDA_GDS_STATS=1` | Log per-tensor transfer timing and throughput |

## Open Questions

- Should this be submitted to llama.cpp upstream first? The `load_all_data()` changes
  and `ggml-cuda` additions are in llama.cpp code that Ollama vendors.
- Should we auto-detect GDS availability and enable it without requiring the env var?
  Current approach is explicit opt-in for safety.
- Should the bounce buffer be pre-allocated and reused rather than malloc/free per
  unaligned tensor? Would reduce overhead but adds complexity.
- What's the right minimum tensor size threshold? 64KB is a guess. Benchmarks will tell.
