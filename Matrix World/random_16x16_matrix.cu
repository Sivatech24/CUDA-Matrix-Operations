#include <iostream>
#include <curand_kernel.h>

#define N 16 // 16x16 matrix

__global__ void fillRandomMatrix(int *matrix, unsigned long seed) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;

        // Initialize CURAND
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Random int between 0 and 99
        matrix[idx] = curand(&state) % 100;
    }
}

__global__ void displayMatrix(int *matrix) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        printf("%2d ", matrix[idx]);
        if (col == N - 1) printf("\n");
    }
}

int main() {
    int *d_matrix;

    // Allocate device memory
    cudaMalloc(&d_matrix, sizeof(int) * N * N);

    // Launch kernel to fill matrix with random numbers
    dim3 threadsPerBlock(N, N);
    fillRandomMatrix<<<1, threadsPerBlock>>>(d_matrix, time(NULL));
    cudaDeviceSynchronize();

    std::cout << "Random 16x16 Matrix:\n";
    displayMatrix<<<1, threadsPerBlock>>>(d_matrix);
    cudaDeviceSynchronize();

    cudaFree(d_matrix);
    return 0;
}
