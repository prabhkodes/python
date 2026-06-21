# Python HPC

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
