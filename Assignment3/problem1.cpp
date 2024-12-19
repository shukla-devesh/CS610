#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <x86intrin.h>
#include <stdlib.h>
#include <fstream>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (512)
#define ALIGNMENT (32)

void matmul_seq(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4_unaligned(float** A, float** B, float** C) {
  __m128 a_line, b_line, c_line;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j += 4){
      c_line = _mm_setzero_ps();
      for(int k = 0; k < N; k++){
        a_line = _mm_set1_ps(A[i][k]);
        b_line = _mm_loadu_ps(&B[k][j]);
        c_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), c_line);
      }
      _mm_storeu_ps(&C[i][j], c_line);
    }
  }
}

void matmul_sse4_aligned(float** A, float** B, float** C) {
  __m128 a_line, b_line, c_line;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j += 4){
      c_line = _mm_setzero_ps();
      for(int k = 0; k < N; k++){
        a_line = _mm_set1_ps(A[i][k]);
        b_line = _mm_load_ps(&B[k][j]);
        c_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), c_line);
      }
      _mm_store_ps(&C[i][j], c_line);
    }
  }
}

void matmul_avx2_unaligned(float** A, float** B, float** C) {
  __m256 a_line, b_line, c_line;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j += 8){
      c_line = _mm256_setzero_ps();
      for(int k = 0; k < N; k++){
        a_line = _mm256_set1_ps(A[i][k]);
        b_line = _mm256_loadu_ps(&B[k][j]);
        c_line = _mm256_add_ps(_mm256_mul_ps(a_line, b_line), c_line);
      }
      _mm256_storeu_ps(&C[i][j], c_line);
    }
  }
}

void matmul_avx2_aligned(float** A, float** B, float** C) {
  __m256 a_line, b_line, c_line;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j += 8){
      c_line = _mm256_setzero_ps();
      for(int k = 0; k < N; k++){
        a_line = _mm256_set1_ps(A[i][k]);
        b_line = _mm256_load_ps(&B[k][j]);
        c_line = _mm256_add_ps(_mm256_mul_ps(a_line, b_line), c_line);
      }
      _mm256_store_ps(&C[i][j], c_line);
    }
  }
}

void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  // auto** A = new float*[N];
  // for (int i = 0; i < N; i++) {
  //   A[i] = new float[N]();
  // }
  // auto** B = new float*[N];
  // for (int i = 0; i < N; i++) {
  //   B[i] = new float[N]();
  // }

  // auto** C_seq = new float*[N];
  // auto** C_sse4 = new float*[N];
  // auto** C_avx2 = new float*[N];
  // for (int i = 0; i < N; i++) {
  //   C_seq[i] = new float[N]();
  //   C_sse4[i] = new float[N]();
  //   C_avx2[i] = new float[N]();
  // }

  float** A = new float*[N];
  float** B_unaligned = new float*[N];
  float** B_aligned = new float*[N];
  float** C_seq = new float*[N];
  float** C_sse4_unaligned = new float*[N];
  float** C_sse4_aligned = new float*[N];
  float** C_avx2_unaligned = new float*[N];
  float** C_avx2_aligned = new float*[N];

  for (int i = 0; i < N; i++) {
    A[i] = static_cast<float*>(_mm_malloc(N * sizeof(float), ALIGNMENT));
    B_unaligned[i] = new float[N]();
    B_aligned[i] = static_cast<float*>(_mm_malloc(N * sizeof(float), ALIGNMENT));
    C_seq[i] = static_cast<float*>(_mm_malloc(N * sizeof(float), ALIGNMENT));
    C_sse4_unaligned[i] = new float[N]();
    C_sse4_aligned[i] = static_cast<float*>(_mm_malloc(N * sizeof(float), ALIGNMENT));
    C_avx2_unaligned[i] = new float[N]();
    C_avx2_aligned[i] = static_cast<float*>(_mm_malloc(N * sizeof(float), ALIGNMENT));
  }

  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0.1F;
      B_aligned[i][j] = 0.2F;
      B_unaligned[i][j] = 0.2F;
      C_seq[i][j] = 0.0F;
      C_sse4_aligned[i][j] = 0.0F;
      C_sse4_unaligned[i][j] = 0.0F;
      C_avx2_aligned[i][j] = 0.0F;
      C_avx2_unaligned[i][j] = 0.0F;
    }
  }

  std::ofstream results("results_prob1.txt", std::ios::app);

  HRTimer start = HR::now();
  matmul_seq(A, B_unaligned, C_seq);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq time: " << duration << " ms" << endl;
  results << duration << ", ";

  start = HR::now();
  matmul_sse4_unaligned(A, B_unaligned, C_sse4_unaligned);
  end = HR::now();
  check_result(C_seq, C_sse4_unaligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4(unaligned) time: " << duration << " ms" << endl;
  results << duration << ", ";

  start = HR::now();
  matmul_sse4_aligned(A, B_aligned, C_sse4_aligned);
  end = HR::now();
  check_result(C_seq, C_sse4_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4(aligned) time: " << duration << " ms" << endl;
  results << duration << ", ";

  start = HR::now();
  matmul_avx2_unaligned(A, B_unaligned, C_avx2_unaligned);
  end = HR::now();
  check_result(C_seq, C_avx2_unaligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2(unaligned) time: " << duration << " ms" << endl;
  results << duration << ", ";

  start = HR::now();
  matmul_avx2_aligned(A, B_aligned, C_avx2_aligned);
  end = HR::now();
  check_result(C_seq, C_avx2_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2(aligned) time: " << duration << " ms" << endl;
  results << duration << endl;

  return EXIT_SUCCESS;
}
