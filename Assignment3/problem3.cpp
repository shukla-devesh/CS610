#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <pthread.h>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <omp.h>

using std::atomic_int;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::list;
using std::ofstream;
using std::strerror;
using std::string;


std::queue<std::string> shared_queue; 
std::mutex mtx;
std::condition_variable cond_var_producer, cond_var_consumer;
bool done = false;

void producer_func(std::ifstream &inputFile, int L, int T, int M, int N){ 
    while(1){
        std::vector<std::string> lines;
        std::string line;

        bool fileReadDone = false;

        #pragma omp critical
        {
            int linesRead = 0;
            while(linesRead < L && std::getline(inputFile, line)){
                lines.push_back(line);
                linesRead++;
            }
            if(linesRead == 0) {
                fileReadDone = true;
            }
        }
        
        if(fileReadDone) return;

        #pragma omp critical(shared_queue_write)
        {
            for (const auto& ln : lines) {
                std::unique_lock<std::mutex> lock(mtx);
                cond_var_producer.wait(lock, [M]{return shared_queue.size() < M;});  // Wait if shared_queue is full
                shared_queue.push(ln);
                cond_var_consumer.notify_one();  // Notify consumer that there is data available in the shared queue
            }
        }
    }
}

void consumer_func(const std::string& outputFilePath) {
    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << outputFilePath << std::endl;
        return;
    }
    while (!done || !shared_queue.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cond_var_consumer.wait(lock, [] { return !shared_queue.empty() || done; });
        while (!shared_queue.empty()) {
            std::string line = shared_queue.front();
            shared_queue.pop();
            lock.unlock();

            outputFile << line << std::endl;

            lock.lock();
            cond_var_producer.notify_one();  // Notify producers that there's space in the shared_queue
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input file> <T> <L> <M> <output file>" << std::endl;
        return 1;
    }

    std::string inputFilePath = argv[1];
    int T = std::stoi(argv[2]);
    int L = std::stoi(argv[3]);
    int M = std::stoi(argv[4]);  // Set shared_queue size
    std::string outputFilePath = argv[5];

    int N = 0;
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << inputFilePath << std::endl;
        return 1;
    }
    
    int producersCompleted = 0;

    #pragma omp parallel num_threads(T + 1)
    {
        int threadId = omp_get_thread_num();
        if(threadId < T){
            producer_func(std::ref(inputFile), L, T, M, N);
        }
        else{
            consumer_func(std::ref(outputFilePath));
        }
        #pragma omp critical
        {
            producersCompleted++;
            if(producersCompleted == T){
                std::unique_lock<std::mutex> lock(mtx);
                done = true;
                cond_var_consumer.notify_one();
            }
        }
    }

    inputFile.close();

    return 0;
}
