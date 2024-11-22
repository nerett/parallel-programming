#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <mutex>
#include <future>
#include <string>
#include <atomic>

std::mutex mtx;

std::atomic<bool> found(false);

bool isDelimiter(char ch)
{
    return std::isspace(ch) || ch == '\0' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v';
}

void searchInBlock(const std::string& filename, const std::string& word, std::promise<bool>&& promise, std::streampos startPos, std::streamsize blockSize)
{
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Thread: unable to open file: " + filename);
        }

        file.seekg(startPos);
        std::vector<char> buffer(blockSize + 100);

        file.read(buffer.data(), buffer.size());
        std::string block(buffer.begin(), buffer.end());

        size_t pos = block.find(word);
        while (pos != std::string::npos) {
            if (found.load()) {
                promise.set_value(false);
                return;
            }

            bool validStart = (pos == 0) || isDelimiter(block[pos - 1]);
            bool validEnd = (pos + word.size() >= block.size()) || isDelimiter(block[pos + word.size()]);

            if (validStart && validEnd) {
                found.store(true);
                promise.set_value(true);
                return;
            }

            pos = block.find(word, pos + 1);
        }

        promise.set_value(false);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cerr << "Exception in thread: " << e.what() << std::endl;
        promise.set_exception(std::current_exception());
    }
}

int main()
{
    const std::string filename = "benchmark3.txt";
    const std::string word = "SEARCHTARGET";
    const int numThreads = 16;

    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    std::streamsize fileSize = file.tellg();
    file.close();

    std::vector<std::thread> threads;
    std::vector<std::future<bool>> futures;
    std::streamsize blockSize = fileSize / numThreads;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        std::promise<bool> promise;
        futures.push_back(promise.get_future());

        std::streampos startPos = i * blockSize;
        threads.emplace_back(searchInBlock, filename, word, std::move(promise), startPos, blockSize);
    }

    bool result = false;
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
        try {
            if (futures[i].get()) {
                result = true;
                break;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception while processing result: " << e.what() << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << (result ? "Found word!" : "Word not found.") << std::endl;
    std::cout << "Search time: " << duration.count() << std::endl;

    return 0;
}
