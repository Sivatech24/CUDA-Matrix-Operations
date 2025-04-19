#include <iostream>
#include <curand_kernel.h>
#include <algorithm>

#define N 1024  // Matrix size: 1024x1024
#define MATRIX_SIZE (N * N)  // Total number of elements

// Kernel to shuffle the numbers and assign them to the matrix
__global__ void fillUniqueRandomMatrix(int *matrix, unsigned long seed, int *randomNumbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE) {
        // Assign a unique number from the shuffled list
        matrix[idx] = randomNumbers[idx];
    }
}

__global__ void displayMatrix(int *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < MATRIX_SIZE) {
        int col = idx % N;
        printf("%7d ", matrix[idx]);
        if (col == N - 1) printf("\n");
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    int *d_matrix, *d_randomNumbers;
    int *randomNumbers = new int[MATRIX_SIZE];
    int *shuffledRandomNumbers = new int[MATRIX_SIZE];

    // Initialize the random number list (0 to MATRIX_SIZE-1)
    for (int i = 0; i < MATRIX_SIZE; i++) {
        randomNumbers[i] = i;
    }

    // Shuffle the random numbers to ensure unique random assignment
    std::random_shuffle(randomNumbers, randomNumbers + MATRIX_SIZE);

    // Allocate device memory for matrix and random number list
    cudaMalloc(&d_matrix, MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_randomNumbers, MATRIX_SIZE * sizeof(int));

    // Copy the shuffled random numbers to device memory
    cudaMemcpy(d_randomNumbers, randomNumbers, MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to fill matrix with shuffled unique random numbers
    fillUniqueRandomMatrix<<<256, 256>>>(d_matrix, time(NULL), d_randomNumbers);
    cudaDeviceSynchronize();

    std::cout << "Random 1024x1024 Matrix with unique values from 0 to " << MATRIX_SIZE - 1 << " at random positions:\n";
    displayMatrix<<<256, 256>>>(d_matrix);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_randomNumbers);

    // Free host memory
    delete[] randomNumbers;
    delete[] shuffledRandomNumbers;

    return 0;
}
