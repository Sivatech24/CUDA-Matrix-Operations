#include <iostream>
#include <curand_kernel.h>

#define N 124  // Matrix size: 124x124
#define MATRIX_SIZE (N * N)  // Total number of elements

__global__ void fillRandomMatrix(int *matrix, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Create a random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < MATRIX_SIZE) {
        int randomIndex = curand(&state) % MATRIX_SIZE;  // Random index from 0 to MATRIX_SIZE-1
        int randomValue = curand(&state) % MATRIX_SIZE;  // Random number from 0 to MATRIX_SIZE-1

        // Fill the matrix with random numbers at random locations
        matrix[randomIndex] = randomValue;
    }
}

__global__ void displayMatrix(int *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < MATRIX_SIZE) {
        int col = idx % N;
        printf("%5d ", matrix[idx]);
        if (col == N - 1) printf("\n");
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    int *d_matrix;

    // Allocate device memory for matrix
    cudaMalloc(&d_matrix, MATRIX_SIZE * sizeof(int));

    // Launch kernel to fill matrix with random numbers
    fillRandomMatrix<<<32, 256>>>(d_matrix, time(NULL));
    cudaDeviceSynchronize();

    std::cout << "Random 124x124 Matrix with values from 0 to " << MATRIX_SIZE - 1 << " at random positions:\n";
    displayMatrix<<<32, 256>>>(d_matrix);
    cudaDeviceSynchronize();

    cudaFree(d_matrix);
    return 0;
}
