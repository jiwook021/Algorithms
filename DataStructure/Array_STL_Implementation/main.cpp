/**
 * @file main.cpp
 * @brief Driver for Array STL Implementation
 */

#include "Array.hpp"
#include <iostream>
#include <algorithm>

int main() {
    Array<int, 5> arr = {5, 3, 1, 4, 2};

    std::cout << "Before sort: ";
    for (auto v : arr) std::cout << v << " ";
    std::cout << "\n";

    std::sort(arr.begin(), arr.end());

    std::cout << "After sort:  ";
    for (auto v : arr) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Size: " << arr.Size() << "\n";
    std::cout << "Front: " << arr.Front() << ", Back: " << arr.Back() << "\n";

    return 0;
}
