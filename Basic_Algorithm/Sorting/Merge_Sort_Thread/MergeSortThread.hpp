/**
 * @file MergeSortThread.hpp
 * @brief Parallel merge sort using std::thread / std::async
 * @details Recursive merge sort that spawns async workers for sub-array sorting.
 *          Falls back to serial sort below a configurable threshold.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <future>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>

namespace MergeSortThread {

struct Person {
    std::string Name;
    int Age;
};

inline std::ostream& operator<<(std::ostream& os, const Person& p) {
    return os << p.Name << "(" << p.Age << ")";
}

template <typename RandomIt, typename Compare>
void MySort(RandomIt first, RandomIt last, Compare comp, int depth = 0) {
    auto size = std::distance(first, last);
    if (size <= 1) return;

    auto mid = first + size / 2;

    if (size > 10000 && depth < 3 && std::thread::hardware_concurrency() > 1) {
        auto future = std::async(std::launch::async,
                                 [&]() { MySort(first, mid, comp, depth + 1); });
        MySort(mid, last, comp, depth + 1);
        future.get();
    } else {
        MySort(first, mid, comp, depth + 1);
        MySort(mid, last, comp, depth + 1);
    }

    std::inplace_merge(first, mid, last, comp);
}

template <typename RandomIt>
void MySort(RandomIt first, RandomIt last) {
    MySort(first, last, std::less<>{});
}

template <typename T>
void PrintVector(const std::vector<T>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& v : vec) std::cout << v << " ";
    std::cout << std::endl;
}

} // namespace MergeSortThread
