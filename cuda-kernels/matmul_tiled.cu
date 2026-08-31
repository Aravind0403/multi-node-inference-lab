#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define TILE 16

// -----------------------------------------------------------------------
// Naive matmul: C = A x B, each thread computes one output element,
// re-reading directly from global memory on every step of the dot product.
// -----------------------------------------------------------------------
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------
// Tiled matmul: each block cooperatively loads TILE x TILE chunks of A and B
// into shared memory once, then every thread in the block reuses that tile
// instead of re-hitting global memory per k.
//
// TODO(you): fill in the two marked sections below.
// -----------------------------------------------------------------------
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = N / TILE;  // assumes N is an exact multiple of TILE for now

    for (int t = 0; t < numTiles; t++) {
        // TODO 1: each thread loads ONE element of A's tile and ONE element of B's tile.
        // Think in terms of: this block covers output rows [blockIdx.y*TILE, +TILE)
        // and output cols [blockIdx.x*TILE, +TILE). For tile step `t`, A's column
        // range and B's row range both slide by `t * TILE` along the shared (k) dimension.
        //
        // tileA[threadIdx.y][threadIdx.x] = A[/* ??? */];
        // tileB[threadIdx.y][threadIdx.x] = B[/* ??? */];

        __syncthreads();  // wait until the whole tile is loaded before anyone reads it

        // TODO 2: accumulate this tile's contribution to `sum`, reading from
        // tileA/tileB (shared memory) instead of A/B (global memory).
        //
        // for (int k = 0; k < TILE; k++) {
        //     sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        // }

        __syncthreads();  // wait until everyone's done reading before the tile gets overwritten
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------
// Host code: run both kernels on the same random NxN input, verify they
// agree, and report median-of-K timings for each (lesson from vector_add:
// single-sample GPU timing is unreliable).
// -----------------------------------------------------------------------
float benchmarkKernel(void (*launch)(const float*, const float*, float*, int, dim3, dim3),
                       const float* d_A, const float* d_B, float* d_C, int N,
                       dim3 grid, dim3 block, int numRuns) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> times(numRuns);

    for (int i = 0; i < numRuns; i++) {
        cudaEventRecord(start);
        launch(d_A, d_B, d_C, N, grid, block);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times.begin(), times.end());
    return (numRuns % 2 == 1)
        ? times[numRuns / 2]
        : (times[numRuns / 2 - 1] + times[numRuns / 2]) / 2.0f;
}

void launchNaive(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmulNaive<<<grid, block>>>(A, B, C, N);
}

void launchTiled(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmulTiled<<<grid, block>>>(A, B, C, N);
}

int main() {
    const int N = 512;  // must be a multiple of TILE (16) for this skeleton
    const size_t bytes = (size_t)N * N * sizeof(float);
    const int NUM_RUNS = 10;

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_naive = (float*)malloc(bytes);
    float* h_C_tiled = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    // --- naive ---
    float naive_ms = benchmarkKernel(launchNaive, d_A, d_B, d_C, N, grid, block, NUM_RUNS);
    cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- tiled ---
    float tiled_ms = benchmarkKernel(launchTiled, d_A, d_B, d_C, N, grid, block, NUM_RUNS);
    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- verify tiled matches naive ---
    bool ok = true;
    for (int i = 0; i < N * N; i++) {
        if (fabsf(h_C_naive[i] - h_C_tiled[i]) > 1e-2f) { ok = false; break; }
    }

    printf("N = %d, TILE = %d\n", N, TILE);
    printf("Naive median time (ms): %f\n", naive_ms);
    printf("Tiled median time (ms): %f\n", tiled_ms);
    printf("Speedup (naive/tiled):  %fx\n", naive_ms / tiled_ms);
    printf("Correctness: %s\n", ok ? "MATCH" : "MISMATCH");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
    return 0;
}
