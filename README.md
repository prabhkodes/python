# Python HPC

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![mpi4py](https://img.shields.io/badge/mpi4py-364d6e?style=flat-square&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?style=flat-square&logoColor=white)
![CuPy](https://img.shields.io/badge/CuPy-76B900?style=flat-square&logo=nvidia&logoColor=white)
![pybind11](https://img.shields.io/badge/pybind11-00599C?style=flat-square&logo=cplusplus&logoColor=white)

Python implementations of parallel and GPU-accelerated HPC patterns. Covers distributed parallelism with `mpi4py`, JIT compilation with Numba, GPU arrays with CuPy, and calling optimised C++ solvers from Python via pybind11.

## Projects

### game_of_life
Conway's Game of Life implemented three ways — NumPy serial, MPI distributed, and Numba JIT/NJIT/stencil benchmarks. Demonstrates how the same stencil problem scales from a single vectorised kernel to a distributed multi-process simulation.

### pybind11_jacobi
2D Jacobi heat diffusion solver built as a progression from serial C++ to distributed MPI+OpenMP to GPU (CuPy), with the C++ versions exposed to Python via pybind11. Includes strong scaling benchmarks across 1 to 16 nodes on Leonardo Booster.

## Dependencies

- Python 3.11+
- `numpy`, `matplotlib`, `mpi4py`, `numba`, `cupy`
- pybind11 (for jacobi serial/parallel)
- OpenMPI
