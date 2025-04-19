#include <iostream>
#include <curand_kernel.h>

__global__ void fillRandom01(int *matrix, unsigned long seed) {
    // Setup RNG
    curandState state;
    curand_init(seed, 0, 0, &state);

    // Generate 0 or 1
    int randVal = curand(&state) % 2;
    matrix[0] = randVal;
}

int main() {
    int *d_matrix, *h_matrix;
    h_matrix = new int[1];

    cudaMalloc(&d_matrix, sizeof(int));

    do {
        // Fill matrix with random 0 or 1
        fillRandom01<<<1, 1>>>(d_matrix, time(NULL));
        cudaMemcpy(h_matrix, d_matrix, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        std::cout << "Generated value: " << h_matrix[0] << std::endl;
    } while (h_matrix[0] == 0);

    std::cout << "Matrix is now non-zero: " << h_matrix[0] << std::endl;

    cudaFree(d_matrix);
    delete[] h_matrix;
    return 0;
}
