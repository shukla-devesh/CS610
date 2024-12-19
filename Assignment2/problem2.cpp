#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <atomic>
#include <barrier>

std::queue<std::string> shared_queue; 
std::mutex mtx;
std::condition_variable cond_var_producer, cond_var_consumer;
// std::atomic<bool> done(false);
bool done = false;

std::mutex inputFile_mutex;
std::mutex shared_queue_write_mutex;


void producer_func(int threadId, std::ifstream &inputFile, int L, int T, int M, int N, std::barrier<>& thread_barrier){
    thread_barrier.arrive_and_wait();    
    while(1){
        inputFile_mutex.lock();
        std::vector<std::string> lines;
        std::string line;
        int i = 0;
        while(i < L && std::getline(inputFile, line)){
            lines.push_back(line);
            i++;
        }
        if(i == 0) {
            inputFile_mutex.unlock();
            return;
        }
        inputFile_mutex.unlock();

        shared_queue_write_mutex.lock();
        for (const auto& ln : lines) {
            std::unique_lock<std::mutex> lock(mtx);
            // std::cout << "Tid = " << threadId << " { " << shared_queue.size() << "   ";
            cond_var_producer.wait(lock, [M]{return shared_queue.size() < M;});  // Wait if shared_queue is full
            // std::cout << " " << shared_queue.size() << " } " << std::endl;
            shared_queue.push(ln);
            // std::cout << ln << std::endl;
            cond_var_consumer.notify_one();  // Notify consumer that there is data available in the shared queue
        }
        shared_queue_write_mutex.unlock();
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
    
    std::barrier thread_barrier(T);

    // Launch producer threads
    std::vector<std::thread> producers;
    for (int i = 0; i < T; ++i) {
        producers.emplace_back(producer_func, i, std::ref(inputFile), L, T, M, N, std::ref(thread_barrier));
    }

    // Launch consumer thread
    std::thread consumerThread(consumer_func, std::ref(outputFilePath));

    // Join producer threads
    for (auto& thread : producers) {
        thread.join();
    }
    inputFile.close();

    // Signal the consumer that production is done
    {
        std::unique_lock<std::mutex> lock(mtx);
        done = true;
        cond_var_consumer.notify_one();
    }

    // Join consumer thread
    consumerThread.join();

    return 0;
}
