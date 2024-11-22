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

void searchInBlock(const std::string& filename, const std::string& word, std::promise<bool>&& promise, std::streampos startPos, std::streamsize blockSize)
{
    try {
        std::cout << "Thread launched!" << std::endl;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Thread: unable to open file: " + filename);
        }

        file.seekg(startPos);
        std::vector<char> buffer(blockSize);
        file.read(buffer.data(), blockSize);

        std::string block(buffer.begin(), buffer.end());


        if (found.load()) {
            promise.set_value(false);
            return;
        }

        if (block.find(word) != std::string::npos) {
            found.store(true);
            promise.set_value(true);
            return;
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
    const std::string filename = "war-and-peace.txt";
    const std::string word = "SEARCHTARGET";
    const int numThreads = 16;

    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    // Получение размера файла
    std::streamsize fileSize = file.tellg();
    file.close();

    std::vector<std::thread> threads;
    std::vector<std::future<bool>> futures;
    std::streamsize blockSize = fileSize / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        std::promise<bool> promise;
        futures.push_back(promise.get_future());
        std::streampos startPos = i * blockSize;
        threads.emplace_back(searchInBlock, filename, word, std::move(promise), startPos, blockSize);
    }


    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }

    // Проверка результатов
    for (auto& future : futures) {
        try {
            if (future.get()) {
                std::cout << "Found word!" << std::endl;
                return 0;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception on result processing: " << e.what() << std::endl;
        }
    }

    std::cout << "Word not found." << std::endl;
    return 0;
}
