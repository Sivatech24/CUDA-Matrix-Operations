# CUDA-Matrix-Operations
This repository contains CUDA-based matrix computation projects, including implementations of matrix multiplication, addition, and optimizations using GPU parallelism for high-performance computing.

---

# CUDA Matrix Operations with Random Unique Values

This repository showcases CUDA-powered matrix operations on square matrices of various sizes:  
**1×1, 2×2, 4×4, 8×8, 16×16, 32×32, 64×64, 128×128, 256×256, 512×512, and 1024×1024**.  
Each matrix is initialized with **random and unique numbers**, and operations such as matrix multiplication are accelerated using CUDA cores for high performance.

## Features

- CUDA-based parallel matrix multiplication  
- Unique random number generation for matrix initialization  
- Performance testing across multiple matrix sizes  
- Scalable and modular code structure  

## Matrix Sizes

Implemented and tested sizes:
```
1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024
```

## Requirements

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- C++ Compiler (e.g., `nvcc`)

## How to Run

Compile and run using:
```bash
nvcc -o matrix matrix.cu
./matrix
```

---

Let me know if you want to include timing results, memory usage, or visual outputs!
