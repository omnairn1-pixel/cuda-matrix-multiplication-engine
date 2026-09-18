#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

// Matrix dimension: N x N
const int N = 1024;

// 1. CUDA Kernel (Device)
__global__ void matrixMulCUDA(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corrected line

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 2. Sequential CPU Baseline (Host)
void matrixMulCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Allocate Host memory
    vector<float> h_A(N * N);
    vector<float> h_B(N * N);
    vector<float> h_C_CPU(N * N);
    vector<float> h_C_GPU(N * N);

    // Initialize matrices with sample values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f;
        h_B[i] = static_cast<float>(rand() % 100) / 10.0f;
    }

    cout << "Matrix Size: " << N << " x " << N << "\n";
    cout << "----------------------------------------\n";

    // --- Benchmark CPU Baseline ---
    cout << "Running CPU baseline...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A.data(), h_B.data(), h_C_CPU.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    cout << "CPU Execution Time: " << cpu_duration.count() << " ms\n";

    // --- Benchmark GPU Acceleration ---
    // Allocate Device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from Host to Device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Configure 2D Grid and Block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // CUDA timing events
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cout << "Running CUDA kernel...\n";
    cudaEventRecord(gpu_start);
    matrixMulCUDA<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    float gpu_duration = 0.0f;
    cudaEventElapsedTime(&gpu_duration, gpu_start, gpu_stop);
    cout << "GPU Kernel Execution Time: " << gpu_duration << " ms\n";

    // Copy result back from Device to Host
    cudaMemcpy(h_C_GPU.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // --- Calculate Speedup and Verify ---
    cout << "----------------------------------------\n";
    cout << "Speedup: " << (cpu_duration.count() / gpu_duration) << "x\n";

    // Numerical validation
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C_CPU[i] - h_C_GPU[i]) > 1e-2) {
            correct = false;
            break;
        }
    }
    cout << "Validation: " << (correct ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
