#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (64);
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_DIM = 8;
const uint32_t INTRA_BLOCK_SKIP = 8;


__global__ void kernel1(const double* in, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1){
    out[i * N * N + j * N + k] = 0.8 * (in[(i - 1) * N * N + j * N + k] + in[(i + 1) * N * N + j * N + k] +
                                        in[i * N * N + (j - 1) * N + k] + in[i * N * N + (j + 1) * N + k] + 
                                        in[i * N * N + j * N + k - 1] + in[i * N * N + j * N + k + 1]);
  }
}

__global__ void kernel2(const double* in, double* out) {
    __shared__ double tile[BLOCK_DIM + 2][TILE_DIM + 2 + 1][BLOCK_DIM + 2];

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;   
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    for(int i = 0; i < BLOCK_DIM; i += INTRA_BLOCK_SKIP){
      int i_global = x + i;
      int j_global = y;
      int k_global = z;

      tile[tx + 1 + i][ty + 1][tz + 1] = in[i_global * N * N + j_global * N + k_global];

      if(tx == 0 && i_global > 0 && i == 0){
        tile[0][ty + 1][tz + 1] = in[(i_global - 1) * N * N + j_global * N + k_global];
      }
      if(tx == INTRA_BLOCK_SKIP - 1 && i_global < N - 1 && i + INTRA_BLOCK_SKIP >= BLOCK_DIM){
        tile[BLOCK_DIM + 1][ty + 1][tz + 1] = in[(i_global + 1) * N * N + j_global * N + k_global];
      }
      if(ty == 0 && j_global > 0){
        tile[tx + 1 + i][0][tz + 1] = in[i_global * N * N + (j_global - 1) * N + k_global];
      }
      if(ty == TILE_DIM - 1 && j_global < N - 1){
        tile[tx + 1 + i][TILE_DIM + 1][tz + 1] = in[i_global * N * N + (j_global + 1) * N + k_global];
      }
      if(tz == 0 && k_global > 0){
        tile[tx + 1 + i][ty + 1][0] = in[i_global * N * N + j_global * N + k_global - 1];
      }
      if(tz == BLOCK_DIM - 1 && k_global < N - 1){
        tile[tx + 1 + i][ty + 1][BLOCK_DIM + 1] = in[i_global * N * N + j_global * N + k_global + 1];
      }
    }
    __syncthreads();
    for(int i = 0; i < BLOCK_DIM; i += INTRA_BLOCK_SKIP){
      int i_global = x + i;
      int j_global = y;
      int k_global = z;
      if(i_global < N - 1 && j_global < N - 1 && k_global < N - 1 && i_global > 0 && j_global > 0 && k_global > 0){
        out[i_global * N * N + j_global * N + k_global] = 0.8 * (tile[tx + i][ty + 1][tz + 1] + tile[tx + 2 + i][ty + 1][tz + 1] +
                                        tile[tx + 1 + i][ty][tz + 1] + tile[tx + 1 + i][ty + 2][tz + 1] + 
                                        tile[tx + 1 + i][ty + 1][tz] + tile[tx + 1 + i][ty + 1][tz + 2]);
      }
    }
}

__global__ void kernel2_opt(const double* in, double* out){
   __shared__ double tile[BLOCK_DIM + 2][TILE_DIM + 2][BLOCK_DIM + 2];

  int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;   
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int i_global = x;
  int j_global = y;
  int k_global = z;
  int i = 0;
  tile[tx + 1 + i][ty + 1][tz + 1] = in[i_global * N * N + j_global * N + k_global];
  if(tx == 0 && i_global > 0 && i == 0){
    tile[0][ty + 1][tz + 1] = in[(i_global - 1) * N * N + j_global * N + k_global];
  }
  if(tx == 2 - 1 && i_global < N - 1 && i + 2 >= BLOCK_DIM){
    tile[BLOCK_DIM + 1][ty + 1][tz + 1] = in[(i_global + 1) * N * N + j_global * N + k_global];
  }
  if(ty == 0 && j_global > 0){
    tile[tx + 1 + i][0][tz + 1] = in[i_global * N * N + (j_global - 1) * N + k_global];
  }
  if(ty == TILE_DIM - 1 && j_global < N - 1){
    tile[tx + 1 + i][TILE_DIM + 1][tz + 1] = in[i_global * N * N + (j_global + 1) * N + k_global];
  }
  if(tz == 0 && k_global > 0){
    tile[tx + 1 + i][ty + 1][0] = in[i_global * N * N + j_global * N + k_global - 1];
  }
  if(tz == BLOCK_DIM - 1 && k_global < N - 1){
    tile[tx + 1 + i][ty + 1][BLOCK_DIM + 1] = in[i_global * N * N + j_global * N + k_global + 1];
  }

  i = 2;
  i_global = x + i;
  j_global = y;
  k_global = z;
  tile[tx + 1 + i][ty + 1][tz + 1] = in[i_global * N * N + j_global * N + k_global];
  if(tx == 0 && i_global > 0 && i == 0){
    tile[0][ty + 1][tz + 1] = in[(i_global - 1) * N * N + j_global * N + k_global];
  }
  if(tx == 2 - 1 && i_global < N - 1 && i + 2 >= BLOCK_DIM){
    tile[BLOCK_DIM + 1][ty + 1][tz + 1] = in[(i_global + 1) * N * N + j_global * N + k_global];
  }
  if(ty == 0 && j_global > 0){
    tile[tx + 1 + i][0][tz + 1] = in[i_global * N * N + (j_global - 1) * N + k_global];
  }
  if(ty == TILE_DIM - 1 && j_global < N - 1){
    tile[tx + 1 + i][TILE_DIM + 1][tz + 1] = in[i_global * N * N + (j_global + 1) * N + k_global];
  }
  if(tz == 0 && k_global > 0){
    tile[tx + 1 + i][ty + 1][0] = in[i_global * N * N + j_global * N + k_global - 1];
  }
  if(tz == BLOCK_DIM - 1 && k_global < N - 1){
    tile[tx + 1 + i][ty + 1][BLOCK_DIM + 1] = in[i_global * N * N + j_global * N + k_global + 1];
  }
  i = 4;
  i_global = x + i;
  j_global = y;
  k_global = z;
  tile[tx + 1 + i][ty + 1][tz + 1] = in[i_global * N * N + j_global * N + k_global];
  if(tx == 0 && i_global > 0 && i == 0){
    tile[0][ty + 1][tz + 1] = in[(i_global - 1) * N * N + j_global * N + k_global];
  }
  if(tx == 2 - 1 && i_global < N - 1 && i + 2 >= BLOCK_DIM){
    tile[BLOCK_DIM + 1][ty + 1][tz + 1] = in[(i_global + 1) * N * N + j_global * N + k_global];
  }
  if(ty == 0 && j_global > 0){
    tile[tx + 1 + i][0][tz + 1] = in[i_global * N * N + (j_global - 1) * N + k_global];
  }
  if(ty == TILE_DIM - 1 && j_global < N - 1){
    tile[tx + 1 + i][TILE_DIM + 1][tz + 1] = in[i_global * N * N + (j_global + 1) * N + k_global];
  }
  if(tz == 0 && k_global > 0){
    tile[tx + 1 + i][ty + 1][0] = in[i_global * N * N + j_global * N + k_global - 1];
  }
  if(tz == BLOCK_DIM - 1 && k_global < N - 1){
    tile[tx + 1 + i][ty + 1][BLOCK_DIM + 1] = in[i_global * N * N + j_global * N + k_global + 1];
  }
  i = 6;
  i_global = x + i;
  j_global = y;
  k_global = z;
  tile[tx + 1 + i][ty + 1][tz + 1] = in[i_global * N * N + j_global * N + k_global];
  if(tx == 0 && i_global > 0 && i == 0){
    tile[0][ty + 1][tz + 1] = in[(i_global - 1) * N * N + j_global * N + k_global];
  }
  if(tx == 2 - 1 && i_global < N - 1 && i + 2 >= BLOCK_DIM){
    tile[BLOCK_DIM + 1][ty + 1][tz + 1] = in[(i_global + 1) * N * N + j_global * N + k_global];
  }
  if(ty == 0 && j_global > 0){
    tile[tx + 1 + i][0][tz + 1] = in[i_global * N * N + (j_global - 1) * N + k_global];
  }
  if(ty == TILE_DIM - 1 && j_global < N - 1){
    tile[tx + 1 + i][TILE_DIM + 1][tz + 1] = in[i_global * N * N + (j_global + 1) * N + k_global];
  }
  if(tz == 0 && k_global > 0){
    tile[tx + 1 + i][ty + 1][0] = in[i_global * N * N + j_global * N + k_global - 1];
  }
  if(tz == BLOCK_DIM - 1 && k_global < N - 1){
    tile[tx + 1 + i][ty + 1][BLOCK_DIM + 1] = in[i_global * N * N + j_global * N + k_global + 1];
  }

  __syncthreads();

  i = 0;
  i_global = x + i;
  j_global = y;
  k_global = z;
  if(i_global < N - 1 && j_global < N - 1 && k_global < N - 1 && i_global > 0 && j_global > 0 && k_global > 0)
    out[i_global * N * N + j_global * N + k_global] = 0.8 * (tile[tx + i][ty + 1][tz + 1] + tile[tx + 2 + i][ty + 1][tz + 1] +
                                                             tile[tx + 1 + i][ty][tz + 1] + tile[tx + 1 + i][ty + 2][tz + 1] + 
                                                             tile[tx + 1 + i][ty + 1][tz] + tile[tx + 1 + i][ty + 1][tz + 2]);

  i = 2;
  i_global = x + i;
  j_global = y;
  k_global = z;
  if(i_global < N - 1 && j_global < N - 1 && k_global < N - 1 && i_global > 0 && j_global > 0 && k_global > 0)
    out[i_global * N * N + j_global * N + k_global] = 0.8 * (tile[tx + i][ty + 1][tz + 1] + tile[tx + 2 + i][ty + 1][tz + 1] +
                                                             tile[tx + 1 + i][ty][tz + 1] + tile[tx + 1 + i][ty + 2][tz + 1] + 
                                                             tile[tx + 1 + i][ty + 1][tz] + tile[tx + 1 + i][ty + 1][tz + 2]);
  i = 4;
  i_global = x + i;
  j_global = y;
  k_global = z;
  if(i_global < N - 1 && j_global < N - 1 && k_global < N - 1 && i_global > 0 && j_global > 0 && k_global > 0)
    out[i_global * N * N + j_global * N + k_global] = 0.8 * (tile[tx + i][ty + 1][tz + 1] + tile[tx + 2 + i][ty + 1][tz + 1] +
                                                             tile[tx + 1 + i][ty][tz + 1] + tile[tx + 1 + i][ty + 2][tz + 1] + 
                                                             tile[tx + 1 + i][ty + 1][tz] + tile[tx + 1 + i][ty + 1][tz + 2]);
  i = 6;
  i_global = x + i;
  j_global = y;
  k_global = z;
  if(i_global < N - 1 && j_global < N - 1 && k_global < N - 1 && i_global > 0 && j_global > 0 && k_global > 0)
    out[i_global * N * N + j_global * N + k_global] = 0.8 * (tile[tx + i][ty + 1][tz + 1] + tile[tx + 2 + i][ty + 1][tz + 1] +
                                                             tile[tx + 1 + i][ty][tz + 1] + tile[tx + 1 + i][ty + 2][tz + 1] + 
                                                             tile[tx + 1 + i][ty + 1][tz] + tile[tx + 1 + i][ty + 1][tz + 2]);
}


__host__ void stencil(const double* in, double* out){
  for(int i = 1; i < N - 1; i++) {
    for(int j = 1; j < N - 1; j++) {
      for(int k = 1; k < N - 1; k++) {
        out[i * N * N + j * N + k] = 0.8 * (in[(i - 1) * N * N + j * N + k] + in[(i + 1) * N * N + j * N + k] +
                                            in[i * N * N + (j - 1) * N + k] + in[i * N * N + (j + 1) * N + k] + 
                                            in[i * N * N + j * N + k - 1] + in[i * N * N + j * N + k + 1]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 1; i < size - 1; i++) {
    for (uint64_t j = 1; j < size - 1; j++) {
      for (uint64_t k = 1; k < size - 1; k++) {
        double this_diff =
            fabs(w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k]);
        if (this_diff > THRESHOLD) {
          numdiffs++;
          if(numdiffs < 100){
            cout << "Diff found at i = " << i << ", j = " << j << ", k = " << k << "; Ref = " << w_ref[i + N * j + N * N * k] << "; Opt = " << w_opt[i + N * j + N * N * k] << "\n";
          }
          if (this_diff - maxdiff > THRESHOLD) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;

  double* in = new double[SIZE];
  double* out_ref = new double[SIZE];
  double* out_naive = new double[SIZE];
  double* out_shared_mem = new double[SIZE];
  double* out_shared_mem_opt = new double[SIZE];

  int MAX_VAL = 100000;
  for(int i = 0; i < SIZE; i++) {
    in[i] = rand() % MAX_VAL;
    out_ref[i] = 0;
    out_naive[i] = 0;
    out_shared_mem[i] = 0;
    out_shared_mem_opt[i] = 0;
  }

  double clkbegin = rtclock();
  stencil(in, out_ref);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;


  // Naive CUDA version
  double *d_in, *d_out;
  cudaCheckError(cudaMalloc(&d_in, SIZE * sizeof(double)));
  cudaCheckError(cudaMalloc(&d_out, SIZE * sizeof(double)));

  cudaEvent_t start_outer, end_outer, start_inner, end_inner;
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_in, in, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((N + blockDim.x - 1)/ blockDim.x , (N + blockDim.y - 1)/ blockDim.y, (N + blockDim.z - 1)/ blockDim.z);

  cudaCheckError(cudaEventRecord(start_inner));
  kernel1<<<gridDim, blockDim>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaMemcpy(out_naive, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  check_result(out_ref, out_naive, N);

  float kernel_time_without_copy, kernel_time_with_copy;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_without_copy, start_inner, end_inner));
  cudaCheckError(cudaEventElapsedTime(&kernel_time_with_copy, start_outer, end_outer));
  std::cout << "Kernel 1 time excluding copy time b/w CPU and GPU (ms): " << kernel_time_without_copy << "\n";
  std::cout << "Kernel 1 time including copy time b/w CPU and GPU (ms): " << kernel_time_with_copy << "\n";


  // Shared memory version
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_in, in, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  blockDim = {INTRA_BLOCK_SKIP, TILE_DIM, BLOCK_DIM};
  gridDim = {(int)(N + BLOCK_DIM - 1)/(BLOCK_DIM) , (int)(N + blockDim.y - 1)/ blockDim.y, (int)(N + blockDim.z - 1)/ blockDim.z};

  cudaCheckError(cudaEventRecord(start_inner));
  kernel2<<<gridDim, blockDim>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaMemcpy(out_shared_mem, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  check_result(out_ref, out_shared_mem, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time_without_copy, start_inner, end_inner));
  cudaCheckError(cudaEventElapsedTime(&kernel_time_with_copy, start_outer, end_outer));
  std::cout << "Kernel 2 time excluding copy time b/w CPU and GPU (ms): " << kernel_time_without_copy << "\n";
  std::cout << "Kernel 2 time including copy time b/w CPU and GPU (ms): " << kernel_time_with_copy << "\n";


  // Optimised shared memory version
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_in, in, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  blockDim = {2, TILE_DIM, BLOCK_DIM};
  gridDim = {(int)(N + BLOCK_DIM - 1)/(BLOCK_DIM) , (int)(N + blockDim.y - 1)/ blockDim.y, (int)(N + blockDim.z - 1)/ blockDim.z};

  cudaCheckError(cudaEventRecord(start_inner));
  kernel2_opt<<<gridDim, blockDim>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaMemcpy(out_shared_mem_opt, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  check_result(out_ref, out_shared_mem_opt, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time_without_copy, start_inner, end_inner));
  cudaCheckError(cudaEventElapsedTime(&kernel_time_with_copy, start_outer, end_outer));
  std::cout << "Optimised kernel 2 time excluding copy time b/w CPU and GPU (ms): " << kernel_time_without_copy << "\n";
  std::cout << "Optimised kernel 2 time including copy time b/w CPU and GPU (ms): " << kernel_time_with_copy << "\n";


  // Pinned memory version
  double *h_in_pinned, *h_out_pinned;
  cudaCheckError(cudaHostAlloc(&h_in_pinned, SIZE * sizeof(double), cudaHostAllocDefault));
  cudaCheckError(cudaHostAlloc(&h_out_pinned, SIZE * sizeof(double), cudaHostAllocDefault));
  for(int i = 0; i < SIZE; i++) {
    h_in_pinned[i] = in[i];
    h_out_pinned[i] = 0;
  }
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_in, h_in_pinned, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  blockDim = {2, TILE_DIM, BLOCK_DIM};
  gridDim = {(int)(N + BLOCK_DIM - 1)/(BLOCK_DIM) , (int)(N + blockDim.y - 1)/ blockDim.y, (int)(N + blockDim.z - 1)/ blockDim.z};

  cudaCheckError(cudaEventRecord(start_inner));
  kernel2_opt<<<gridDim, blockDim>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaMemcpy(h_out_pinned, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  check_result(out_ref, h_out_pinned, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time_without_copy, start_inner, end_inner));
  cudaCheckError(cudaEventElapsedTime(&kernel_time_with_copy, start_outer, end_outer));
  std::cout << "Pinned memory kernel time excluding copy time b/w CPU and GPU (ms): " << kernel_time_without_copy << "\n";
  std::cout << "Pinned memory kernel time including copy time b/w CPU and GPU (ms): " << kernel_time_with_copy << "\n";
  
  cudaCheckError(cudaFreeHost(h_in_pinned));
  cudaCheckError(cudaFreeHost(h_out_pinned));

  // UVM version
  double *h_in_uvm, *h_out_uvm;
  cudaCheckError(cudaMallocManaged(&h_in_uvm, SIZE * sizeof(double)));
  cudaCheckError(cudaMallocManaged(&h_out_uvm, SIZE * sizeof(double)));
  for(int i = 0; i < SIZE; i++) {
    h_in_uvm[i] = in[i];
    h_out_uvm[i] = 0;
  }
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_in, h_in_uvm, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  blockDim = {2, TILE_DIM, BLOCK_DIM};
  gridDim = {(int)(N + BLOCK_DIM - 1)/(BLOCK_DIM) , (int)(N + blockDim.y - 1)/ blockDim.y, (int)(N + blockDim.z - 1)/ blockDim.z};

  cudaCheckError(cudaEventRecord(start_inner));
  kernel2_opt<<<gridDim, blockDim>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaMemcpy(h_out_uvm, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  check_result(out_ref, h_out_uvm, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time_without_copy, start_inner, end_inner));
  cudaCheckError(cudaEventElapsedTime(&kernel_time_with_copy, start_outer, end_outer));
  std::cout << "UVM kernel time excluding copy time b/w CPU and GPU (ms): " << kernel_time_without_copy << "\n";
  std::cout << "UVM kernel time including copy time b/w CPU and GPU (ms): " << kernel_time_with_copy << "\n";

  cudaCheckError(cudaEventDestroy(start_outer));
  cudaCheckError(cudaEventDestroy(end_outer));
  cudaCheckError(cudaEventDestroy(start_inner));
  cudaCheckError(cudaEventDestroy(end_inner));
  delete[] in;
  delete[] out_ref;
  delete[] out_naive;
  delete[] out_shared_mem;
  delete[] out_shared_mem_opt;
  cudaCheckError(cudaFree(d_in));
  cudaCheckError(cudaFree(d_out));
  cudaCheckError(cudaFree(h_in_uvm));
  cudaCheckError(cudaFree(h_out_uvm));

  return EXIT_SUCCESS;
}