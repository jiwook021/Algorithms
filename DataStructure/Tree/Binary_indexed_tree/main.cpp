/**
 * @file main.cpp
 * @brief Driver for BinaryIndexedTree (Fenwick Tree)
 */

#include "BinaryIndexedTree.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<long long> values = {1, 3, 5, 7, 9, 11};
    BinaryIndexedTree bit(values);

    std::cout << "Array: 1 3 5 7 9 11 (1-indexed)\n";
    std::cout << "PrefixSum(3) = " << bit.PrefixSum(3) << "\n";
    std::cout << "RangeSum(2,5) = " << bit.RangeSum(2, 5) << "\n";

    bit.Update(3, 5);
    std::cout << "After Update(3, +5):\n";
    std::cout << "PrefixSum(3) = " << bit.PrefixSum(3) << "\n";

    return 0;
}
