#include <atomic>
#include <memory>
#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <filesystem>
#include "concurrentStack.h"

#define TOTAL_CONCURRENT_OPS 10000000                               // Change this to test with different number of concurrent operations
#define NUM_THREADS 16                                       // Change this to test with different number of threads
#define OPS_PER_THREAD (TOTAL_CONCURRENT_OPS)/(NUM_THREADS)       // Change this to test with different number of operations per thread

using namespace std;
using namespace std::chrono;
using HR = high_resolution_clock;
using HRTimer = HR::time_point;
using std::filesystem::path;

void read_data(path pth, uint64_t n, uint32_t* data) {
  FILE* fptr = fopen(pth.string().c_str(), "rb");
  string fname = pth.string();
  if (!fptr) {
    string error_msg = "Unable to open file: " + fname;
    perror(error_msg.c_str());
  }
  int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
  if (freadStatus == 0) {
    string error_string = "Unable to read the file " + fname;
    perror(error_string.c_str());
  }
  fclose(fptr);
}

// Test function with random operations
void test_stack(LockFreeStack& stack, int num_operations, uint32_t* data) {
    std::srand(std::time(nullptr));

    for (int i = 0; i < num_operations; ++i) {
        if (std::rand() % 2 == 0) {
            int value = data[i];
            stack.push(value);
            // std::cout << "Pushed: " << value << "\n";
            // stack.printStack();
        } else {
            int result = stack.pop();
            // if (result >= 0)
                // std::cout << "Popped: " << result << "\n";
            // else
                // std::cout << "Pop failed: stack empty\n";
            // stack.printStack();
        }
    }
}

int main() {
    LockFreeStack stack;

    path cwd = std::filesystem::current_path();
    path path_insert_values = cwd / "random_values_insert.bin";
    int num_values = TOTAL_CONCURRENT_OPS;
    auto* tmp_values_insert = new uint32_t[num_values];
    read_data(path_insert_values, num_values, tmp_values_insert);

    std::vector<std::thread> threads;
    HRTimer start = HR::now();
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(test_stack, std::ref(stack), OPS_PER_THREAD, tmp_values_insert + i * OPS_PER_THREAD);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    HRTimer end = HR::now();
    duration<double> time_span = duration_cast<duration<double>>(end - start);
    std::cout << "Time taken: " << time_span.count() << " seconds\n";

    return 0;
}
