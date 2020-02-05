# speedup-cuda

Try to be closer to implementation of GEMM in cublas.

Now it is naive CUDA C, and reach 90% performance of cublas (sgemm_128x128x8_NN_vec) on GTX 960M (Compute Capability 5.0)
