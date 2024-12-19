#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <papi.h>

using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)
// #define ITERS (5)


// void handle_error(int retval){
//   cout << "handle_error called\nPAPI error " << retval << " " << ": " << PAPI_strerror(retval) << endl ;
//   exit(EXIT_FAILURE);
// }

void matmul_ijk(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, int a_blksz, int b_blksz, int c_blksz, const int n) {
  for (int i = 0; i < n; i += a_blksz){
    for (int j = 0; j < n; j += b_blksz){
      for (int k = 0; k < n; k += c_blksz){
      /* B×B mini -matrix (blocks) multiplications */
        for (int i1 = i; i1 < i + a_blksz; i1++){
          for (int j1 = j; j1 < j + b_blksz; j1++){
            for (int k1 = k; k1 < k + c_blksz; k1++){
              C[i1*n+j1] += A[i1*n + k1]*B[k1*n + j1];
            }
          }
        }
      }
    }
  }
}

void init(uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      mat[i * SIZE + j] = 1;
    }
  }
}

void print_matrix(const uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << mat[i * SIZE + j] << "\t";
    }
    cout << "\n";
  }
}

void check_result(const uint32_t *ref, const uint32_t *opt, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (ref[i * SIZE + j] != opt[i * SIZE + j]) {
        assert(false && "Diff found between sequential and blocked versions!\n");
      }
    }
  }
}

// void check_event_support(int event_code) {
//     if (PAPI_query_event(event_code) != PAPI_OK) {
//         fprintf(stderr, "Event code %d is not supported on this hardware.\n", event_code);
//         exit(1);
//     }
//     cout << "CHECKED SUPPORT FOR " << event_code << endl;
// }

int main(int argc, char ** argv) {
  if(argc != 5){
    // Mode = 0 for sequential and 1 for blocking version
    std::cerr << "Usage : <File> <BlockSize for A> <BlockSize for B> <BlockSize for C> <Mode>" << endl; 
    return 1;
  }
  int a_blksz = stoi(argv[1]);
  int b_blksz = stoi(argv[2]);
  int c_blksz = stoi(argv[3]);
  int mode = stoi(argv[4]);

  if((N % a_blksz) || (N % b_blksz) || (N % c_blksz)){
    std::cerr << "All block sizes must be divisors of " << N << std::endl;
    return 1;
  }

  if(mode != 0 && mode != 1){
    std::cerr << "Enter valid value for mode (0 for sequential and 1 for blocking version)" << endl;
    return 1;
  }

  uint32_t *A = new uint32_t[N * N];
  uint32_t *B = new uint32_t[N * N];
  uint32_t *C_seq, *C_blk;

  if(mode == 0) C_seq = new uint32_t[N * N];

  if(mode == 1) C_blk = new uint32_t[N * N];

  // unsigned long long total_duration_seq = 0;
  // unsigned long long total_duration_blk = 0;
  // unsigned long long total_L1_miss_seq = 0;
  // unsigned long long total_L2_miss_seq = 0;
  // unsigned long long total_L1_miss_blk = 0;
  // unsigned long long total_L2_miss_blk = 0;

  std::ofstream timelog("time_logs_CS610.csv", std::ios::app);
  std::ofstream L1log("L1_logs_CS610.csv", std::ios::app);
  std::ofstream L2log("L2_logs_CS610.csv", std::ios::app);
  // std::ofstream resfile("results_CS610.csv", std::ios::app);

  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    cerr << "PAPI library version mismatch: " << retval << " != " << PAPI_VER_CURRENT << "\n";
    exit(EXIT_FAILURE);
  } else if (retval < 0) {
    cerr << "PAPI library initialization error: " << retval << " != " << PAPI_VER_CURRENT << "\n";
    exit(EXIT_FAILURE);
  }

  int eventset = PAPI_NULL;
  retval = PAPI_create_eventset(&eventset);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_create_eventset()" << endl;
    exit(EXIT_FAILURE);
  }

  // check_event_support(PAPI_L1_DCM);
  // check_event_support(PAPI_L2_DCM);


  if (PAPI_add_event(eventset, PAPI_L1_DCM) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_L1_DCM!\n";
    exit(EXIT_FAILURE);
  }
  if (PAPI_add_event(eventset, PAPI_L2_DCM) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_L2_DCM!\n";
    exit(EXIT_FAILURE);
  }

  // for(int i = 0; i < ITERS; i++){
    init(A, N);
    init(B, N);
    if(mode == 0){
      init(C_seq, N);

      retval = PAPI_start(eventset);
      if (PAPI_OK != retval) {
        cerr << "Error at PAPI_start()" << endl;
        exit(EXIT_FAILURE);
      }

      HRTimer start = HR::now();
      matmul_ijk(A, B, C_seq, N);
      HRTimer end = HR::now();

      long long int values[2];
      retval = PAPI_stop(eventset, values);
      if(PAPI_OK != retval){
        cerr << "Error at PAPI_stop()" << endl;
        exit(EXIT_FAILURE);
      }

      auto duration = duration_cast<microseconds>(end - start).count();

      cout << "Time without blocking (us): " << duration << "\n";
      cout << values[0] << " " << values[1] << endl;
      timelog << duration << std::endl;
      L1log << values[0] << endl;
      L2log << values[1] << endl; 
      delete[] A;
      delete[] B;
      delete[] C_seq;
    }
    else{

      init(C_blk, N);

      retval = PAPI_start(eventset);
      if (PAPI_OK != retval) {
        cerr << "Error at PAPI_start()" << endl;
        exit(EXIT_FAILURE);
      }

      HRTimer start = HR::now();
      matmul_ijk_blocking(A, B, C_blk, a_blksz, b_blksz, c_blksz, N);
      HRTimer end = HR::now();

      long long int values2[2];
      retval = PAPI_stop(eventset, values2);
      if(PAPI_OK != retval){
        cerr << "Error at PAPI_stop()" << endl;
        exit(EXIT_FAILURE);
      }

      auto duration = duration_cast<microseconds>(end - start).count();

      cout << "Time with blocking (us): " << duration << "\n";
      cout << values2[0] << " " << values2[1] << endl;
      timelog << duration << ", " << a_blksz << ", " << b_blksz << ", " << c_blksz << std::endl;
      L1log << values2[0] << ", " << a_blksz << ", " << b_blksz << ", " << c_blksz << endl;
      L2log << values2[1] << ", " << a_blksz << ", " << b_blksz << ", " << c_blksz << endl;
      delete[] A;
      delete[] B;
      delete[] C_blk;
    }
    // check_result(C_seq, C_blk, N);
  // }

  PAPI_cleanup_eventset(eventset);
  PAPI_destroy_eventset(&eventset);
  PAPI_shutdown();

  // unsigned long long avg_duration_seq = total_duration_seq ;
  // unsigned long long avg_duration_blk = total_duration_blk ;
  // unsigned long long avg_L1_miss_seq = total_L1_miss_seq;
  // unsigned long long avg_L2_miss_seq = total_L2_miss_seq;
  // unsigned long long avg_L1_miss_blk = total_L1_miss_blk;
  // unsigned long long avg_L2_miss_blk = total_L2_miss_blk;
  // resfile << "SEQ, " << avg_duration_seq << std::endl;
  // resfile << "SEQ, " << avg_L1_miss_seq << ", " << avg_L2_miss_seq << endl;
  // resfile << "BLK, " << avg_duration_blk << ", " << a_blksz << ", " << b_blksz << ", " << c_blksz << std::endl;
  // resfile << "BLK, " << avg_L1_miss_blk << ", " << avg_L2_miss_blk << endl;

  return EXIT_SUCCESS;
}
