#include <iostream>
#include <curand_kernel.h>

#define N 64  // Matrix size: 64x64

__global__ void fillRandomMatrix(int *matrix, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N * N) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        matrix[idx] = curand(&state) % 100;  // Random number between 0–99
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void displayMatrix(int *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N * N) {
        int col = idx % N;  // Calculate column
        printf("%2d ", matrix[idx]);
        if (col == N - 1) printf("\n");
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    int *d_matrix;

    // Allocate device memory
    cudaMalloc(&d_matrix, N * N * sizeof(int));

    // Launch kernel to fill matrix with random numbers
    fillRandomMatrix<<<32, 256>>>(d_matrix, time(NULL));
    cudaDeviceSynchronize();

    std::cout << "Random 64x64 Matrix:\n";
    displayMatrix<<<32, 256>>>(d_matrix);
    cudaDeviceSynchronize();

    cudaFree(d_matrix);
    return 0;
}
