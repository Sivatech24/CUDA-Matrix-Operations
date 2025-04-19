#include <iostream>
#include <curand_kernel.h>

#define N 32  // 32x32 matrix

__global__ void fillRandomMatrix(int *matrix, unsigned long seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;

        curandState state;
        curand_init(seed, idx, 0, &state);

        matrix[idx] = curand(&state) % 100;  // Random number 0–99
    }
}

__global__ void displayMatrix(int *matrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        printf("%2d ", matrix[idx]);
        if (col == N - 1) printf("\n");
    }
}

int main() {
    int *d_matrix;

    size_t size = N * N * sizeof(int);
    cudaMalloc(&d_matrix, size);

    dim3 threadsPerBlock(16, 16);   // 256 threads per block
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);  // 2x2 blocks for 32x32

    // Fill matrix
    fillRandomMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix, time(NULL));
    cudaDeviceSynchronize();

    std::cout << "Random 32x32 Matrix:\n";
    displayMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix);
    cudaDeviceSynchronize();

    cudaFree(d_matrix);
    return 0;
}
