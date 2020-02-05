#include <stdio.h>
#include <cuda.h>


#define N 1024
#define THREADX 16
#define THREADY 16
#define K_BLKSIZE 32
#define MA 2
#define MB 2
#define BS 16
#define KBS 8


__global__ void MatMul_v1(float *A, float *B, float *C)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  float tmp = 0.0f;
  for (int i = 0; i < N; i++) {
    tmp += A[row*N + i] * B[i*N + col];
  }
  C[row*N + col] = tmp;
}


__global__ void MatMul_v2(float *A, float *B, float *C)
{
  int blkCol = blockIdx.x;
  int blkRow = blockIdx.y;
  int col = threadIdx.x;
  int row = threadIdx.y;
  float tmp = 0.0f;
  __shared__ float A_blk[THREADY][K_BLKSIZE];
  __shared__ float B_blk[K_BLKSIZE][THREADX];

  for (int blk = 0; blk < N / K_BLKSIZE; blk++) {
    for (int e = 0; e < K_BLKSIZE / THREADX; e++) {
      A_blk[row][e*THREADX + col] = A[blkRow*blockDim.y*N + blk*K_BLKSIZE + row*N + e*THREADX + col];
    }
    for (int e = 0; e < K_BLKSIZE / THREADY; e++) {
      B_blk[e*THREADY + row][col] = B[blk*K_BLKSIZE*N + blkCol*blockDim.x + e*THREADY*N + row*N + col];
    }
    __syncthreads();
    for (int k = 0; k < K_BLKSIZE; k++) {
      tmp += A_blk[row][k] * B_blk[k][col];
    }
    __syncthreads();
  }
  C[blkRow*blockDim.y*N + blkCol*blockDim.x + row*N + col] = tmp;
}


__global__ void MatMul_v3(float *A, float *B, float *C)
{
  int blkidx = blockIdx.x;
  int blkidy = blockIdx.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;


  __shared__ float cacheS[(MA+MB)*BS*BS];
  float *cacheA = cacheS;
  float *cacheB = cacheS + MA*BS*BS;

  float rst[MA][MB] = { { 0.0 } };


  float regA[MA];
  float regB[MB];

  float tmpA[MA];
  float tmpB[MB];


  #pragma  unroll
  for (int blk = 0; blk < N; blk += BS) {
    #pragma unroll
    for (int i = 0; i < MA; i++) {
      regA[i] = A[(blkidy*MA*BS + i*BS + tidy)*N + blk + tidx];
    }
    #pragma unroll
    for (int i = 0; i < MB; i++) {
      regB[i] = B[(blk + tidy)*N + blkidx*MB*BS + i*BS + tidx];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < MA; i++) {
      cacheA[i*BS*BS + tidy*BS + tidx] = regA[i];
    }
    #pragma unroll
    for (int i = 0; i < MB; i++) {
      cacheB[MB*BS*tidy + i*BS + tidx] = regB[i];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < BS; i++) {
      #pragma unroll
      for (int ia = 0; ia < MA; ia++) {
        tmpA[ia] = cacheA[ia*BS*BS + tidy*BS + i];
      }
      #pragma unroll
      for (int ib = 0; ib < MB; ib++) {
        tmpB[ib] = cacheB[MB*BS*i + ib*BS + tidx];
      }
      #pragma unroll
      for (int ia = 0; ia < MA; ia++) {
        #pragma unroll
        for (int ib = 0; ib < MB; ib++) {
          rst[ia][ib] += tmpA[ia]*tmpB[ib];
        }
      }
    }
  }

  #pragma unroll
  for (int ia = 0; ia < MA; ia++) {
    #pragma unroll
    for (int ib = 0; ib < MB; ib++) {
      C[(blkidy*MA*BS+ia*BS+tidy)*N + blkidx*MB*BS + ib*BS + tidx] = rst[ia][ib];
    }
  }
}


__global__ void MatMul_v4(float *A, float *B, float *C)
{
  int blkidx = blockIdx.x;
  int blkidy = blockIdx.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;


  __shared__ float cacheS[(MA+MB)*BS*BS*2];
  float *cacheA = cacheS;
  float *cacheB = cacheS + MA*BS*BS;

  float rst[MA][MB] = { { 0.0 } };


  float regA[MA];
  float regB[MB];

  float tmpA[MA];
  float tmpB[MB];


  // first iter
  #pragma unroll
  for (int i = 0; i < MA; i++) {
    regA[i] = A[(blkidy*MA*BS + i*BS + tidy)*N + tidx];
  }
  #pragma unroll
  for (int i = 0; i < MB; i++) {
    regB[i] = B[tidy*N + blkidx*MB*BS + i*BS + tidx];
  }

  #pragma unroll
  for (int i = 0; i < MA; i++) {
    cacheA[i*BS*BS + tidy*BS + tidx] = regA[i];
  }
  #pragma unroll
  for (int i = 0; i < MB; i++) {
    cacheB[MB*BS*tidy + i*BS + tidx] = regB[i];
  }

  // intermediate iter
  #pragma unroll
  for (int blk = 1; blk < N/BS; blk++) {
    #pragma unroll
    for (int i = 0; i < MA; i++) {
      regA[i] = A[(blkidy*MA*BS + i*BS + tidy)*N + blk*BS + tidx];
    }
    #pragma unroll
    for (int i = 0; i < MB; i++) {
      regB[i] = B[(blk*BS + tidy)*N + blkidx*MB*BS + i*BS + tidx];
    }

    #pragma unroll
    for (int i = 0; i < MA; i++) {
      cacheA[i*BS*BS + tidy*BS + tidx + (blk&1)*(MA+MB)*BS*BS] = regA[i];
    }
    #pragma unroll
    for (int i = 0; i < MB; i++) {
      cacheB[MB*BS*tidy + i*BS + tidx + (blk&1)*(MA+MB)*BS*BS] = regB[i];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < BS; i++) {
      #pragma unroll
      for (int ia = 0; ia < MA; ia++) {
        tmpA[ia] = cacheA[ia*BS*BS + tidy*BS + i + (!(blk&1))*(MA+MB)*BS*BS];
      }
      #pragma unroll
      for (int ib = 0; ib < MB; ib++) {
        tmpB[ib] = cacheB[MB*BS*i + ib*BS + tidx + (!(blk&1))*(MA+MB)*BS*BS];
      }
      #pragma unroll
      for (int ia = 0; ia < MA; ia++) {
        for (int ib = 0; ib < MB; ib++) {
          rst[ia][ib] += tmpA[ia]*tmpB[ib];
        }
      }
    }
  }

  // final iter
  #pragma unroll
  for (int i = 0; i < BS; i++) {
    #pragma unroll
    for (int ia = 0; ia < MA; ia++) {
      tmpA[ia] = cacheA[ia*BS*BS + tidy*BS + i + (!((N/BS)&1))*(MA+MB)*BS*BS];
    }
    #pragma unroll
    for (int ib = 0; ib < MB; ib++) {
      tmpB[ib] = cacheB[MB*BS*i + ib*BS + tidx + (!((N/BS)&1))*(MA+MB)*BS*BS];
    }
    #pragma unroll
    for (int ia = 0; ia < MA; ia++) {
      for (int ib = 0; ib < MB; ib++) {
        rst[ia][ib] += tmpA[ia]*tmpB[ib];
      }
    }
  }

  #pragma unroll
  for (int ia = 0; ia < MA; ia++) {
    #pragma unroll
    for (int ib = 0; ib < MB; ib++) {
      C[(blkidy*MA*BS+ia*BS+tidy)*N + blkidx*MB*BS + ib*BS + tidx] = rst[ia][ib];
    }
  }
}


int main(int argc, char* argv[])
{

  int m = N;
  int n = N;
  int k = N;
  float *a = (float*) malloc(m*k*sizeof(float));
  float *b = (float*) malloc(k*n*sizeof(float));
  float *c = (float*) malloc(m*n*sizeof(float));

  for (int i = 0; i < m*k; i++) {
    a[i] = i;
  }

  for (int i = 0; i < k*n; i++) {
    b[i] = i;
  }

  float *d_a;
  float *d_b;
  float *d_c;

  cudaMalloc((void**)&d_a, m*k*sizeof(*a));
  cudaMalloc((void**)&d_b, k*n*sizeof(*b));
  cudaMalloc((void**)&d_c, m*n*sizeof(*c));

  cudaMemcpy(d_a, a, m*k*sizeof(*a), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k*n*sizeof(*b), cudaMemcpyHostToDevice);

  dim3 threads(THREADX, THREADY);
  //dim3 blocks(N/threads.x, N/threads.y);
  dim3 blocks(N/MA/BS, N/MB/BS);
  MatMul_v5<<<blocks, threads>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, m*n*sizeof(*c), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    fprintf(stderr, "%.2f ", c[i]);
  }
  fprintf(stderr, "\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);

  return 0;
}
