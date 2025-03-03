/**
 * @file main.cpp
 * @brief Driver program for the ThreadSafeHashMap demo.
 *
 * Demonstrates basic operations and a multi-threaded benchmark.
 *
 * Build:  make && ./main
 */

#include "ThreadSafeHashMap.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cassert>

/**
 * @brief Run a multi-threaded benchmark.
 * @param numThreads Number of threads.
 * @param opsPerThread Operations per thread.
 */
static void Benchmark(int numThreads, int opsPerThread) {
    ThreadSafeHashMap<int, std::string> map(32, 0.75f);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&map, i, opsPerThread]() {
            std::mt19937 gen(i);
            std::uniform_int_distribution<> keyDist(1, 1000);
            std::uniform_int_distribution<> opDist(0, 2);
            for (int j = 0; j < opsPerThread; ++j) {
                int key = keyDist(gen);
                switch (opDist(gen)) {
                    case 0: map.Insert(key, "v" + std::to_string(j)); break;
                    case 1: map.Find(key); break;
                    case 2: map.Erase(key); break;
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << numThreads << " threads x " << opsPerThread
              << " ops => " << ms << " ms  (size=" << map.Size()
              << ", lf=" << map.LoadFactor() << ")\n";
}

int main() {
    // Basic operations demo
    ThreadSafeHashMap<int, std::string> map;
    map.Insert(1, "One");
    map.Insert(2, "Two");
    map.Insert(3, "Three");

    assert(map.Find(1).value() == "One");
    assert(map.Find(2).value() == "Two");
    assert(!map.Find(4).has_value());
    assert(map.Size() == 3);

    map.Erase(1);
    assert(!map.Find(1).has_value());

    map.Clear();
    assert(map.Empty());
    std::cout << "Basic operations passed.\n";

    // Benchmarks
    std::cout << "\nBenchmarks:\n";
    Benchmark(1, 10000);
    Benchmark(4, 10000);
    Benchmark(8, 10000);

    std::cout << "All demos completed.\n";
    return 0;
}
