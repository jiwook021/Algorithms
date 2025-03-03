/**
 * @file main.cpp
 * @brief Driver for QuickSort
 */

#include "QuickSort.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {5, 3, 8, 1, 9, 2, 7, 4, 6};

    std::cout << "Before: ";
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    QuickSort(v.begin(), v.end());

    std::cout << "After:  ";
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
