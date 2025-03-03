/**
 * @file main.cpp
 * @brief Driver for QuickDelete (O(1) removal from unordered vectors)
 */

#include "QuickDelete.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v{123, 456, 789, 100, 200};

    std::cout << "Original: ";
    for (int i : v) std::cout << i << " ";
    std::cout << "\n";

    QuickRemoveAt(v, 2u);
    std::cout << "After QuickRemoveAt(v, 2): ";
    for (int i : v) std::cout << i << " ";
    std::cout << "\n";

    QuickRemoveAt(v, std::find(v.begin(), v.end(), 123));
    std::cout << "After QuickRemoveAt(v, find(123)): ";
    for (int i : v) std::cout << i << " ";
    std::cout << "\n";

    return 0;
}
