#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

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

const uint64_t N_2D = (1 << 6);
const int FILTER_SIZE_2D = 5;
const int FILTER_RADIUS_2D = FILTER_SIZE_2D / 2;
const uint64_t TILE_DIM_2D = 16;
const uint64_t TILE_DIM_2D_OPT = 16;
const uint64_t OUT_TILE_DIM_2D = TILE_DIM_2D_OPT - FILTER_SIZE_2D + 1;

const uint64_t MAX_VAL = 100;

const uint64_t N_3D = 128;
const int FILTER_SIZE_3D = 5;
const int FILTER_RADIUS_3D = FILTER_SIZE_3D / 2;
const uint64_t TILE_DIM_3D = 8;
const uint64_t TILE_DIM_3D_OPT = 8;
const uint64_t OUT_TILE_DIM_3D = TILE_DIM_3D_OPT - FILTER_SIZE_3D + 1;

// Basic 2D convolution kernel using global memory
__global__ void kernel2D(const float* input, float* output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N_2D && y < N_2D) {
    float sum = 0.0f;
    int count = 0;
    for (int i = -FILTER_RADIUS_2D; i <= FILTER_RADIUS_2D; ++i) {
      for (int j = -FILTER_RADIUS_2D; j <= FILTER_RADIUS_2D; ++j) {
        int nx = x + i;
        int ny = y + j;
        if (nx >= 0 && nx < N_2D && ny >= 0 && ny < N_2D) {
          sum += input[ny * N_2D + nx];
        }
        count++;
      }
    }
    output[y * N_2D + x] = sum/count;
  }
}

__global__ void kernel2D_opt(const float* input, float* output){
  __shared__ float shared_mem[TILE_DIM_2D_OPT][TILE_DIM_2D_OPT];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.y * OUT_TILE_DIM_2D + ty;
  int col_o = blockIdx.x * OUT_TILE_DIM_2D + tx;
  int row_i = row_o - FILTER_RADIUS_2D;
  int col_i = col_o - FILTER_RADIUS_2D;

  if((row_i >= 0) && (row_i < N_2D) && (col_i >= 0) && (col_i < N_2D))
    shared_mem[ty][tx] = input[row_i * N_2D + col_i];
  else
    shared_mem[ty][tx] = 0.0f;

  __syncthreads();

  float res = 0.0f;
  if(ty < OUT_TILE_DIM_2D && tx < OUT_TILE_DIM_2D){
    int count = 0;
    for(int i = 0; i < FILTER_SIZE_2D; i++){
      for(int j = 0; j < FILTER_SIZE_2D; j++){
        res += shared_mem[i + ty][j + tx];
        count++;
      }
    }
    if(row_o < N_2D && col_o < N_2D){
      output[row_o * N_2D + col_o] = res/count;
    }
  }
}


// Basic 3D convolution kernel using global memory
__global__ void kernel3D(float* input, float* output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < N_3D && y < N_3D && z < N_3D) {
    float sum = 0.0f;
    int count = 0;
    for (int i = -FILTER_RADIUS_3D; i <= FILTER_RADIUS_3D; ++i) {
      for (int j = -FILTER_RADIUS_3D; j <= FILTER_RADIUS_3D; ++j) {
        for (int k = -FILTER_RADIUS_3D; k <= FILTER_RADIUS_3D; ++k) {
          int nx = x + i;
          int ny = y + j;
          int nz = z + k;
          if (nx >= 0 && nx < N_3D && ny >= 0 && ny < N_3D && nz >= 0 && nz < N_3D) {
            sum += input[nz * N_3D * N_3D + ny * N_3D + nx]; 
          }
          count++;
        }
      }
    }
    output[(z * N_3D + y) * N_3D + x] = sum / count;
  }
}

__global__ void kernel3D_opt(const float* input, float* output){
  __shared__ float shared_mem[TILE_DIM_3D_OPT][TILE_DIM_3D_OPT][TILE_DIM_3D_OPT];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int row_o = blockIdx.y * OUT_TILE_DIM_3D + ty;
  int col_o = blockIdx.x * OUT_TILE_DIM_3D + tx;
  int dep_o = blockIdx.z * OUT_TILE_DIM_3D + tz;
  int row_i = row_o - FILTER_RADIUS_3D;
  int col_i = col_o - FILTER_RADIUS_3D;
  int dep_i = dep_o - FILTER_RADIUS_3D;

  if((row_i >= 0) && (row_i < N_3D) && (col_i >= 0) && (col_i < N_3D) && (dep_i >= 0) && (dep_i < N_3D))
    shared_mem[tz][ty][tx] = input[dep_i * N_3D * N_3D + row_i * N_3D + col_i];
  else
    shared_mem[tz][ty][tx] = 0.0f;

  __syncthreads();

  float res = 0.0f;
  if(ty < OUT_TILE_DIM_3D && tx < OUT_TILE_DIM_3D && tz < OUT_TILE_DIM_3D){
    int count = 0;
    for(int i = 0; i < FILTER_SIZE_3D; i++){
      for(int j = 0; j < FILTER_SIZE_3D; j++){
        for(int k = 0; k < FILTER_SIZE_3D; k++){
          res += shared_mem[k + tz][i + ty][j + tx];
          count++;
        }
      }
    }
    if(row_o < N_3D && col_o < N_3D && dep_o < N_3D){
      output[(dep_o * N_3D + row_o) * N_3D + col_o] = res/count;
    }
  }
}

__host__ void convolution_2D_ref(const float* input, float* output) {
  for (int i = 0; i < N_2D; ++i) {
    for (int j = 0; j < N_2D; ++j) {
      float sum = 0.0f;
      int count = 0;
      for (int k = -FILTER_RADIUS_2D; k <= FILTER_RADIUS_2D; ++k) {
        for (int l = -FILTER_RADIUS_2D; l <= FILTER_RADIUS_2D; ++l) {
          int nx = i + k;
          int ny = j + l;
          if (nx >= 0 && nx < N_2D && ny >= 0 && ny < N_2D) {
            sum += input[ny * N_2D + nx];
          }
          count++;
        }
      }
      output[j * N_2D + i] = sum / count;
    }
  }
}

__host__ void convolution_3D_ref(const float* input, float* output) {
  for (int i = 0; i < N_3D; ++i) {
    for (int j = 0; j < N_3D; ++j) {
      for(int k = 0; k < N_3D; ++k){
        float sum = 0.0f;
        int count = 0;
        for (int l = -FILTER_RADIUS_3D; l <= FILTER_RADIUS_3D; ++l) {
          for (int m = -FILTER_RADIUS_3D; m <= FILTER_RADIUS_3D; ++m) {
            for (int n = -FILTER_RADIUS_3D; n <= FILTER_RADIUS_3D; ++n) {
              int nx = i + l;
              int ny = j + m;
              int nz = k + n;
              if (nx >= 0 && nx < N_3D && ny >= 0 && ny < N_3D && nz >= 0 && nz < N_3D) {
                sum += input[nz * N_3D * N_3D + ny * N_3D + nx];
              }
              count++;
            }
          }
        }
        output[k * N_3D * N_3D + j * N_3D + i] = sum / count;
      }
    }
  }
}


__host__ void check_result_3D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N_3D; i++) {
    for (uint64_t j = 0; j < N_3D; j++) {
      for (uint64_t k = 0; k < N_3D; k++) {
        double this_diff = w_ref[i + N_3D * j + N_3D * N_3D * k] - w_opt[i + N_3D * j + N_3D * N_3D * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if(numdiffs < 100) cout << "Diff found at " << i << ", " << j << ", " << k << endl;
          if (this_diff > maxdiff) {
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

__host__ void check_result_2D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N_2D; i++) {
    for (uint64_t j = 0; j < N_2D; j++) {
      double this_diff = w_ref[i + N_2D * j] - w_opt[i + N_2D * j];
      if (std::fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if(numdiffs < 100) cout << "Diff found at " << i << ", " << j << endl;
        if (this_diff > maxdiff) {
          maxdiff = this_diff;
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

void print2D(const float* A) {
  for (int i = 0; i < N_2D; ++i) {
    for (int j = 0; j < N_2D; ++j) {
      cout << A[i * N_2D + j] << "\t";
    }
    cout << "n";
  }
}

void print3D(const float* A) {
  for (int i = 0; i < N_3D; ++i) {
    for (int j = 0; j < N_3D; ++j) {
      for (int k = 0; k < N_3D; ++k) {
        cout << A[i * N_3D * N_3D + j * N_3D + k] << "\t";
      }
      cout << "n";
    }
    cout << "n";
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
  int size_2D = N_2D * N_2D;
  float* h_input2D = static_cast<float*>(malloc(size_2D * sizeof(float)));
  float* h_output2D_ref = static_cast<float*>(malloc(size_2D * sizeof(float)));
  float* h_output2D_basic = static_cast<float*>(malloc(size_2D * sizeof(float)));
  float* h_output2D_opt = static_cast<float*>(malloc(size_2D * sizeof(float)));
  for (int i = 0; i < size_2D; ++i) {
    h_input2D[i] = static_cast<float>(rand() / RAND_MAX);
  }

  double start_host = rtclock();
  convolution_2D_ref(h_input2D, h_output2D_ref);
  double end_host = rtclock();
  float ref_time = (end_host - start_host) * 1000.0;
  std::cout << "Reference 2D convolution time (ms): " << ref_time << "\n";

  // Allocate device memory
  float *d_input2D, *d_output2D;
  cudaCheckError(cudaMalloc(&d_input2D, size_2D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_output2D, size_2D * sizeof(float)));

  // Copy data from host to device
  cudaEvent_t start_outer, end_outer, start_inner, end_inner;
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_input2D, h_input2D, size_2D * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim2D(TILE_DIM_2D, TILE_DIM_2D);
  dim3 gridDim2D((N_2D + blockDim2D.x - 1) / blockDim2D.x, (N_2D + blockDim2D.y - 1) / blockDim2D.y);

  cudaCheckError(cudaEventRecord(start_inner));
  kernel2D<<<gridDim2D, blockDim2D>>>(d_input2D, d_output2D);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  float kernel_time_inner;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_inner, start_inner, end_inner));
  std::cout << "Basic Kernel2D time excluding time for copying between GPU and CPU (ms): " << kernel_time_inner << "\n";

  cudaCheckError(cudaMemcpy(h_output2D_basic, d_output2D, size_2D * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  
  float kernel_time_outer;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_outer, start_outer, end_outer));
  std::cout << "Basic Kernel2D time including time for copying between GPU and CPU (ms): " << kernel_time_outer << "\n";
  check_result_2D((const float*)h_output2D_ref, (const float*)h_output2D_basic);
  

  // Optimised 2D convolution kernel using shared memory
  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_input2D, h_input2D, size_2D * sizeof(float), cudaMemcpyHostToDevice));
  
  blockDim2D = {TILE_DIM_2D_OPT, TILE_DIM_2D_OPT};
  gridDim2D = {(int)(N_2D + blockDim2D.x - 1) / blockDim2D.x, (int)(N_2D + blockDim2D.y - 1) / blockDim2D.y};
  
  cudaCheckError(cudaEventRecord(start_inner));
  kernel2D_opt<<<gridDim2D, blockDim2D>>>(d_input2D, d_output2D);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaEventElapsedTime(&kernel_time_inner, start_inner, end_inner));
  std::cout << "Optimised Kernel2D time excluding time for copying between GPU and CPU (ms): " << kernel_time_inner << "\n";

  cudaCheckError(cudaMemcpy(h_output2D_opt, d_output2D, size_2D * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  
  cudaCheckError(cudaEventElapsedTime(&kernel_time_outer, start_outer, end_outer));
  std::cout << "Optimised Kernel2D time including time for copying between GPU and CPU (ms): " << kernel_time_outer << "\n";
  check_result_2D((const float*)h_output2D_ref, (const float*)h_output2D_opt);




  // 3D convolution
  int size_3D = N_3D * N_3D * N_3D;
  float* h_input3D = static_cast<float*>(malloc(size_3D * sizeof(float)));
  float* h_output3D_ref = static_cast<float*>(malloc(size_3D * sizeof(float)));
  float* h_output3D_basic = static_cast<float*>(malloc(size_3D * sizeof(float)));
  float* h_output3D_opt = static_cast<float*>(malloc(size_3D * sizeof(float)));
  
  for (int i = 0; i < size_3D; ++i) {
    h_input3D[i] = static_cast<float>(rand() / RAND_MAX);
  }

  start_host = rtclock();
  convolution_3D_ref(h_input3D, h_output3D_ref);
  end_host = rtclock();
  ref_time = (end_host - start_host) * 1000.0;
  std::cout << "Reference 3D convolution time (ms): " << ref_time << "\n";

  // Allocate device memory
  float *d_input3D, *d_output3D;
  cudaCheckError(cudaMalloc(&d_input3D, size_3D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_output3D, size_3D * sizeof(float)));

  // Copy data from host to device
  cudaCheckError(cudaEventCreate(&start_outer));
  cudaCheckError(cudaEventCreate(&end_outer));
  cudaCheckError(cudaEventCreate(&start_inner));
  cudaCheckError(cudaEventCreate(&end_inner));

  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_input3D, h_input3D, size_3D * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim3D(TILE_DIM_3D, TILE_DIM_3D, TILE_DIM_3D);
  dim3 gridDim3D((N_3D + blockDim3D.x - 1) / blockDim3D.x, (N_3D + blockDim3D.y - 1) / blockDim3D.y, (N_3D + blockDim3D.z - 1) / blockDim3D.z);

  cudaCheckError(cudaEventRecord(start_inner));
  kernel3D<<<gridDim3D, blockDim3D>>>(d_input3D, d_output3D);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaEventElapsedTime(&kernel_time_inner, start_inner, end_inner));
  std::cout << "Basic Kernel3D time excluding time for copying between GPU and CPU (ms): " << kernel_time_inner << "\n";

  cudaCheckError(cudaMemcpy(h_output3D_basic, d_output3D, size_3D * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  
  cudaCheckError(cudaEventElapsedTime(&kernel_time_outer, start_outer, end_outer));
  std::cout << "Basic Kernel3D time including time for copying between GPU and CPU (ms): " << kernel_time_outer << "\n";
  check_result_3D((const float*)h_output3D_ref, (const float*)h_output3D_basic);
  

  // Optimised 3D convolution kernel using shared memory
  cudaCheckError(cudaEventRecord(start_outer));
  cudaCheckError(cudaMemcpy(d_input3D, h_input3D, size_3D * sizeof(float), cudaMemcpyHostToDevice));
  
  blockDim3D = {TILE_DIM_3D_OPT, TILE_DIM_3D_OPT, TILE_DIM_3D_OPT};
  gridDim3D = {(int)(N_3D + blockDim3D.x - 1) / blockDim3D.x, (int)(N_3D + blockDim3D.y - 1) / blockDim3D.y};
  
  cudaCheckError(cudaEventRecord(start_inner));
  kernel3D_opt<<<gridDim3D, blockDim3D>>>(d_input3D, d_output3D);
  cudaCheckError(cudaEventRecord(end_inner));
  cudaCheckError(cudaEventSynchronize(end_inner));

  cudaCheckError(cudaEventElapsedTime(&kernel_time_inner, start_inner, end_inner));
  std::cout << "Optimised Kernel3D time excluding time for copying between GPU and CPU (ms): " << kernel_time_inner << "\n";

  cudaCheckError(cudaMemcpy(h_output3D_opt, d_output3D, size_3D * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(end_outer));
  cudaCheckError(cudaEventSynchronize(end_outer));
  
  cudaCheckError(cudaEventElapsedTime(&kernel_time_outer, start_outer, end_outer));
  std::cout << "Optimised Kernel3D time including time for copying between GPU and CPU (ms): " << kernel_time_outer << "\n";
  check_result_3D((const float*)h_output3D_ref, (const float*)h_output3D_opt);

  cudaCheckError(cudaEventDestroy(start_outer));
  cudaCheckError(cudaEventDestroy(start_inner));
  cudaCheckError(cudaEventDestroy(end_outer));
  cudaCheckError(cudaEventDestroy(end_inner));
  
  // Free device memory
  cudaCheckError(cudaFree(d_input2D));
  cudaCheckError(cudaFree(d_output2D));
  cudaCheckError(cudaFree(d_input3D));
  cudaCheckError(cudaFree(d_output3D));

  // Free host memory
  free(h_input2D);
  free(h_output2D_basic);
  free(h_output2D_ref);
  free(h_output2D_opt);
  free(h_input3D);
  free(h_output3D_basic);
  free(h_output3D_ref);
  free(h_output3D_opt);
  return EXIT_SUCCESS;
}
