#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <thrust/scan.h>
#include <chrono>

// const uint64_t CHUNK_SIZE = (1 << 10); // Process data in smaller chunks

const uint64_t NUM_TH_PER_BLCK = 512;
const uint64_t ELE_PER_BLOCK = (2 * NUM_TH_PER_BLCK);
const uint64_t N = (1 << 20);
const uint64_t MAX_VAL = 100;

using std::cerr;
using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::milliseconds;
using std::chrono::microseconds;

#define cudaCheckError(ans)                                                                        \
{ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
		exit(code);
	}
}

__host__ void thrust_sum(const uint32_t *in, uint32_t *out) {
	thrust::exclusive_scan(in, in + N, out);
}

__global__ void cuda_sum(const uint32_t *input, uint32_t *output, uint32_t *block_sum, int n) {
	int blockID = blockIdx.x;
	int thid = threadIdx.x;
	int start = blockID * ELE_PER_BLOCK;
	int end = n;
	__shared__ uint32_t temp[ELE_PER_BLOCK];

	__syncthreads();

	temp[2 * thid] = (start + 2 * thid < end) ? input[start + 2 * thid] : 0;
    temp[2 * thid + 1] = (start + 2 * thid + 1 < end) ? input[start + 2 * thid + 1] : 0;

	int offset = 1;
	for (int d = NUM_TH_PER_BLCK; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

	if (thid == 0) {
		block_sum[blockID] = temp[ELE_PER_BLOCK - 1];
		temp[ELE_PER_BLOCK - 1] = 0;
	}

	for (int d = 1; d < ELE_PER_BLOCK; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        uint32_t t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
	__syncthreads();

	if(start + 2 * thid < end) output[start + (2 * thid)] = temp[2 * thid];
	if(start + 2 * thid + 1 < end) output[start + (2 * thid) + 1] = temp[2 * thid + 1];
	__syncthreads();
}

__global__ void post_cuda_sum(const uint32_t *input, uint32_t *output, uint32_t *sum_offset) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int offset = blockID * ELE_PER_BLOCK;
	output[offset + threadID] = sum_offset[blockID] + input[offset + threadID];
}


__host__ int get_cuda_sum(uint32_t* d_in, uint32_t* d_out, int num_elements){
	int threads_per_block = NUM_TH_PER_BLCK;
	int num_blocks = (num_elements + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;

	uint32_t *block_sum, *sum_offset;
	cudaCheckError(cudaMalloc((void**)&sum_offset, num_blocks * sizeof(uint32_t)));
	cudaCheckError(cudaMalloc((void**)&block_sum, num_blocks * sizeof(uint32_t)));

	cuda_sum<<<num_blocks, threads_per_block>>>(d_in, d_out, block_sum, num_elements);

	if(num_blocks > 1){
		cudaCheckError(cudaMemcpy(d_in, d_out, num_elements * sizeof(uint32_t),cudaMemcpyDeviceToDevice));
		get_cuda_sum(block_sum, sum_offset, num_blocks);
		post_cuda_sum<<<num_blocks, ELE_PER_BLOCK>>>(d_in, d_out, sum_offset);
	}
	cudaCheckError(cudaFree(block_sum));
	cudaCheckError(cudaFree(sum_offset));
	return 0;
}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt, const uint64_t size) {
	uint32_t maxdiff = 0.0, this_diff = 0.0;
	int numdiffs = 0;
	for (uint64_t i = 0; i < size; i++) {
		if(w_ref[i] > w_opt[i]){
			this_diff = w_ref[i] - w_opt[i];
		}
		else{
			this_diff = w_opt[i] - w_ref[i];
		}
		if (this_diff != 0) {
			numdiffs++;
			if (this_diff > maxdiff) {
				maxdiff = this_diff;
			}
		}
	}

	if (numdiffs > 0) {
		cout << numdiffs << " Differences found between the two arrays. Max diff: " << maxdiff << "\n";
	} else {
		cout << "No differences found between base and test versions\n";
	}
}

int main() {
	auto* h_input = new uint32_t[N];
  	for(int i = 0; i < N; i++){
    	h_input[i] = rand() % 100;
  	}

  	// Using Thrust code as reference
  	auto* h_thrust_ref = new uint32_t[N];
  	std::fill_n(h_thrust_ref, N, 0);
	
  	HRTimer local_start = HR::now();
  	thrust_sum(h_input, h_thrust_ref);
  	HRTimer local_end = HR::now();
  	auto duration = duration_cast<microseconds>(local_end - local_start).count();  
  	cout << "Thrust time: (microseconds) " << duration << endl;
	
	// Using CUDA
  	auto* h_cuda = new uint32_t[N];
  	uint32_t* d_input;
  	uint32_t* d_output;
	uint32_t* block_sum;

  	cudaCheckError(cudaMalloc(&d_input, N * sizeof(uint32_t)));
  	cudaCheckError(cudaMalloc(&d_output, N * sizeof(uint32_t)));
	cudaCheckError(cudaMalloc(&block_sum, N * sizeof(uint32_t)));

	cudaCheckError(cudaHostAlloc(&h_cuda, N * sizeof(uint32_t), cudaHostAllocDefault));

  	cudaEvent_t start_inner, start_outer, end_inner, end_outer;
	cudaCheckError(cudaEventCreate(&start_outer));
	cudaCheckError(cudaEventCreate(&end_outer));
	cudaCheckError(cudaEventCreate(&start_inner));
	cudaCheckError(cudaEventCreate(&end_inner));

	cudaCheckError(cudaEventRecord(start_outer));
  	cudaCheckError(cudaMemcpy(d_input, h_input, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  	cudaCheckError(cudaEventRecord(start_inner));

	get_cuda_sum(d_input, d_output, N);

  	cudaCheckError(cudaEventRecord(end_inner));
  	cudaCheckError(cudaEventSynchronize(end_inner));

  	cudaCheckError(cudaMemcpy(h_cuda, d_output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  	cudaCheckError(cudaEventRecord(end_outer));
  	cudaCheckError(cudaEventSynchronize(end_outer));

  	check_result(h_thrust_ref, h_cuda, N);

  	float kernel_time_inner, kernel_time_outer;
  	cudaCheckError(cudaEventElapsedTime(&kernel_time_inner, start_inner, end_inner));
  	cudaCheckError(cudaEventElapsedTime(&kernel_time_outer, start_outer, end_outer));
  	std::cout << "Cuda time excluding time for copying b/w CPU and GPU: (ms) " << kernel_time_inner << "\n";
  	std::cout << "Cuda time including time for copying b/w CPU and GPU: (ms) " << kernel_time_outer << "\n";

  	cudaCheckError(cudaEventDestroy(start_outer));
  	cudaCheckError(cudaEventDestroy(end_outer));
  	cudaCheckError(cudaEventDestroy(start_inner));
	cudaCheckError(cudaEventDestroy(end_inner));

  	cudaCheckError(cudaFree(d_input));
  	cudaCheckError(cudaFree(d_output));
	cudaCheckError(cudaFreeHost(h_cuda));

  	delete[] h_input;
  	delete[] h_thrust_ref;
	return EXIT_SUCCESS;
}