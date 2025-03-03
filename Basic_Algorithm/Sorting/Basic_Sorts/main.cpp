/**
 * @file main.cpp
 * @brief Driver for BasicSorts
 */

#include "BasicSorts.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    BasicSorts::BubbleSort(v.begin(), v.end());

    std::cout << "Sorted: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
