# High-Performance CUDA Matrix Multiplication Engine

A benchmarked C++ and CUDA acceleration engine that computes large-scale matrix multiplication (N x N) in parallel. This project demonstrates low-level hardware optimization, 2D thread hierarchy management, and host-device memory pipelines compared against a single-threaded sequential CPU baseline.

---

## Performance Benchmark

Benchmarks were conducted on an **NVIDIA Tesla T4 GPU** (16 GB VRAM) vs. an **Intel Xeon CPU @ 2.20 GHz** using 1024 x 1024 single-precision floating-point matrices (N = 1024).

| Implementation | Execution Time (ms) | Speedup Factor | Verification |
| :--- | :--- | :--- | :--- |
| Sequential CPU (Baseline) | 3,150.42 ms | 1.0x | Reference |
| Naive CUDA Kernel (GPU) | 18.25 ms | ~172x | PASSED |

*Note: GPU timing captures pure compute kernel execution using hardware events (`cudaEventRecord`). The benchmark demonstrates the elimination of nested loop latency via parallel 2D grid/block thread scheduling.*

---

## Repository Structure
```text
.
├── matrix_mult.cu    # CUDA C++ source code (CPU baseline + GPU kernel + benchmarks)
└── README.md         # Project documentation and performance analysis
```
---

## Architecture and Implementation

### 1. 2D Thread Hierarchy Mapping
Instead of relying on sequential loops, the workload is distributed across hardware threads scheduled in a 2D grid:
* **Block Dimensions:** 16 x 16 threads (256 threads per block, maximizing warp occupancy).
* **Grid Dimensions:** Dynamically calculated to fully cover matrix dimensions without out-of-bounds reads: `dim3 numBlocks((N + 15) / 16, (N + 15) / 16)`.
* **Thread Coordinates:** Each thread calculates its unique 2D position:
  * `row = blockIdx.y * blockDim.y + threadIdx.y`
  * `col = blockIdx.x * blockDim.x + threadIdx.x`

### 2. Host-Device Memory Pipeline
* Contiguous 1D memory buffers allocated on the device using `cudaMalloc` for row-major access (`Index = row * N + col`).
* Explicit Host-to-Device data transfers performed via `cudaMemcpyHostToDevice`.
* Hardware barrier synchronization enforced with `cudaEventSynchronize` before copying results back to host memory via `cudaMemcpyDeviceToHost`.

---

## Build and Run

### Prerequisites
* NVIDIA GPU with Compute Capability 5.0 or higher
* CUDA Toolkit (`nvcc` compiler)
* C++11 or higher

### Compilation
Compile the `.cu` source code with level-3 optimization flags:
```bash
nvcc -O3 matrix_mult.cu -o matrix_mult
```

### Execution
Run the compiled binary:
```bash
./matrix_mult
```

### Sample Terminal Output
```text
Matrix Size: 1024 x 1024
Running CPU baseline...
CPU Execution Time: 3150.42 ms
Running CUDA kernel...
GPU Kernel Execution Time: 18.25 ms
Speedup: 172.6x
Validation: PASSED
```

---

## Numerical Verification
The program performs automated error checking between the CPU reference output and the GPU output buffer:
`max |C_CPU[i] - C_GPU[i]| < 1e-3`

Any deviation beyond `1e-3` triggers an execution failure flag, verifying arithmetic parity between CPU and GPU architectures.