// Coarse-grained locking implies 1 lock for the whole map
// Fine-grained locking implies 1 lock for each key in the map, which is
// encouraged

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <chrono>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;

// Max different files
const int MAX_FILES = 10;
const int MAX_SIZE = 10;
int MAX_THREADS = 5;

struct t_data {
  uint32_t tid;
};

struct word_tracker {
  uint64_t word_count[8*MAX_FILES]; // Padding this array so that elements at offset of 64 bytes (size of an L1 cache line) fall into separate cache lines.
  uint64_t total_lines_processed;   // This reduces false sharing among different elements of array, which were otherwise contiguosly present in the original array.
  uint64_t total_words_processed;
  pthread_mutex_t word_count_mutex;
} tracker;

std::queue<std::string> shared_pq;
pthread_mutex_t pq_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t line_count_mutex = PTHREAD_MUTEX_INITIALIZER;

void *thread_runner(void *);

void print_usage(char *prog_name) {
  cerr << "usage: " << prog_name << " <producer count> <input file>\n";
  exit(EXIT_FAILURE);
}

void print_counters() {
  for (int id = 0; id < MAX_THREADS; ++id) {
    std::cout << "Thread " << id << " counter: " << tracker.word_count[8*id]
              << '\n';
  }
}

void fill_producer_buffer(std::string &input) {
  std::fstream input_file;
  input_file.open(input, ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening the top-level input file!" << endl;
    exit(EXIT_FAILURE);
  }

  std::filesystem::path p(input);
  std::string line;
  while (getline(input_file, line)) {
    shared_pq.push(p.parent_path() / line);
  }
}

int thread_count = 0;

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  if (argc != 3) {
    print_usage(argv[0]);
  }

  thread_count = strtol(argv[1], NULL, 10);
  MAX_THREADS = thread_count;
  std::string input = argv[2];
  fill_producer_buffer(input);

  pthread_t threads_worker[thread_count];

  int file_count;

  struct t_data *args_array =
      (struct t_data *)malloc(sizeof(struct t_data) * thread_count);
  for (int i = 0; i < thread_count; i++)
    tracker.word_count[8*i] = 0;
  tracker.total_lines_processed = 0;
  tracker.word_count_mutex = PTHREAD_MUTEX_INITIALIZER;

  for (int i = 0; i < thread_count; i++) {
    args_array[i].tid = i;
    pthread_create(&threads_worker[i], nullptr, thread_runner,
                   (void *)&args_array[i]);
  }

  for (int i = 0; i < thread_count; i++)
    pthread_join(threads_worker[i], NULL);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  print_counters();
  cout << "Total words processed: " << tracker.total_words_processed << "\n";
  cout << "Total line processed: " << tracker.total_lines_processed << "\n";

  cout << "Total time taken: " << duration.count() << " microseconds\n";
  return EXIT_SUCCESS;
}

void *thread_runner(void *th_args) {
  struct t_data *args = (struct t_data *)th_args;
  uint32_t thread_id = args->tid;
  std::fstream input_file;
  std::string fileName;
  std::string line;

  pthread_mutex_lock(&pq_mutex);
  fileName = shared_pq.front();
  shared_pq.pop();
  pthread_mutex_unlock(&pq_mutex);

  input_file.open(fileName.c_str(), ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening input file from a thread!" << endl;
    exit(EXIT_FAILURE);
  }
  int words_counted_locally = 0;
  int lines_processed_locally = 0;

  while (getline(input_file, line)) {
    lines_processed_locally++;    // Modification - To avoid true sharing on the shared variable total_lines_processed
    std::string delimiter = " ";    // so that a thread updates the shared variable only once after it has completed all its computations.
    size_t pos = 0;                 // This also reduces lock contention significantly.
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);
      tracker.word_count[8*thread_id]++;    // Modification - Using padding to avoid false sharing problem for the elements of this array

      words_counted_locally++;  // Modification - To avoid true sharing on the shared variable total_words_processed for same reasons as the 
      line.erase(0, pos + delimiter.length());  // variable total_lines_processed.
    }
  }
  pthread_mutex_lock(&line_count_mutex);
  tracker.total_lines_processed += lines_processed_locally; // Shared variable updated at the end rather than during each iteration.
  pthread_mutex_unlock(&line_count_mutex);

  pthread_mutex_lock(&tracker.word_count_mutex);
  tracker.total_words_processed += words_counted_locally; // Shared variable updated only once at the end, and not during each iteration.
  pthread_mutex_unlock(&tracker.word_count_mutex);

  input_file.close();

  pthread_exit(nullptr);
}
