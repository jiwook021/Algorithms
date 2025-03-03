/**
 * @file main.cpp
 * @brief Driver for UnorderedSet (SimpleUnorderedSet)
 */

#include "UnorderedSet.hpp"
#include <iostream>

int main() {
    SimpleUnorderedSet<int> set;
    set.insert(10);
    set.insert(20);
    set.insert(30);

    std::cout << "Size: " << set.size() << "\n";
    std::cout << "Contains 20: " << set.Contains(20) << "\n";
    std::cout << "Contains 40: " << set.Contains(40) << "\n";

    set.erase(20);
    std::cout << "After erase(20), size: " << set.size() << "\n";
    std::cout << "Contains 20: " << set.Contains(20) << "\n";

    return 0;
}
