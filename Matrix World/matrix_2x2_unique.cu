#include <iostream>
#include <curand_kernel.h>

#define N 2 // 2x2 matrix

__global__ void fillRandomMatrix(int *matrix, unsigned long seed) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;

        // Set up CURAND
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random number between 0 and 99
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

    cudaMalloc(&d_matrix, sizeof(int) * N * N);

    dim3 threadsPerBlock(N, N);

    // Fill with random numbers
    fillRandomMatrix<<<1, threadsPerBlock>>>(d_matrix, time(NULL));
    cudaDeviceSynchronize();

    std::cout << "Random 2x2 Matrix:\n";
    displayMatrix<<<1, threadsPerBlock>>>(d_matrix);
    cudaDeviceSynchronize();

    cudaFree(d_matrix);
    return 0;
}
