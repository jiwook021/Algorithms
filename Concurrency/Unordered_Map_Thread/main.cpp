/**
 * @file main.cpp
 * @brief Driver program for the ThreadSafeLinkedList demo.
 *
 * Demonstrates multithreaded linked list operations with proper
 * synchronization using std::shared_mutex and std::jthread.
 *
 * Build:  make && ./main
 */

#include "ThreadSafeLinkedList.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <random>

/**
 * @brief Thread routine that performs random operations on the list.
 * @param list The shared linked list.
 * @param id Thread identifier.
 * @param iterations Number of operations to perform.
 */
template <typename T>
static void TestThread(ThreadSafeLinkedList<T>& list, int id, int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 5);
    constexpr int kNum = 10;

    for (int i = 0; i < iterations; ++i) {
        switch (distrib(gen)) {
            case 0:
                list.PushFrontAndLog((id + i) % kNum, id);
                break;
            case 1:
                list.PopFrontAndLog(id);
                break;
            case 2:
                if (!list.Empty())
                    list.InsertAfterAndLog((i * i) % kNum, (id + i) % kNum, id);
                else
                    list.PushFrontEmptyAndLog((id * 1000 + i * i * i) % kNum, id);
                break;
            case 3:
                list.ContainsAndLog((id * i) % kNum, id);
                break;
            case 4:
                list.RemoveAndLog((id * 1000 + id) % kNum, id);
                break;
            case 5:
                for (int j = 0; j < 10; ++j)
                    list.PushFrontAndLog((id + i + j) % kNum, id);
                for (int j = 0; j < 11; ++j)
                    list.PopFrontAndLog(id);
                break;
        }
    }
}

int main() {
    ThreadSafeLinkedList<int> list;
    list.PushFront(3);
    list.PushFront(2);
    list.PushFront(1);

    std::cout << "Initial list: " << list.ToString() << "\n\n";

    const int kNumThreads = 10;
    const int kOps = 10;
    std::vector<std::jthread> threads;
    for (int i = 0; i < kNumThreads; ++i)
        threads.emplace_back(TestThread<int>, std::ref(list), i, kOps);

    // std::jthread joins automatically on destruction -- no manual join needed.
    threads.clear();

    std::cout << "Final list: " << list.ToString() << "\n";
    std::cout << "List size: " << list.Size() << "\n";
    return 0;
}
