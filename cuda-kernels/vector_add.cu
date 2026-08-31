% % writefile vector_add.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

        // CUDA kernel to perform vector addition
        __global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 20; // ~1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 1024; // Typical block size
    // Calculate grid size based on N and blockSize
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch the vectorAdd kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool ok = true;
    for (int i = 0; i < N; i++)
    {
        if (h_c[i] != 3.0f)
        {
            ok = false;
            break;
        }
    }
    printf("Result: %s\n", ok ? "CORRECT" : "WRONG");
    printf("Kernel time: %f ms\n", ms);
    printf("Block size: %d, Grid size: %d, Total threads: %lld\n",
           blockSize, gridSize, (long long)blockSize * gridSize);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}