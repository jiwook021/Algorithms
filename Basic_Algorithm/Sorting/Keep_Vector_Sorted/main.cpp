/**
 * @file main.cpp
 * @brief Driver for KeepVectorSorted
 */

#include "KeepVectorSorted.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace KeepVectorSorted;

int main() {
    std::vector<int> v = {1, 3, 5, 7, 9};

    InsertSorted(v, 4);
    InsertSorted(v, 0);
    InsertSorted(v, 10);

    std::cout << "Sorted vector: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
