/**
 * @file main.cpp
 * @brief Driver for SetOps (set union, intersection, difference)
 */

#include "SetOps.hpp"
#include <iostream>

int main() {
    std::set<int> a = {1, 2, 3, 4, 5};
    std::set<int> b = {3, 4, 5, 6, 7};
    std::set<int> result;

    SetUnion(a, b, result);
    std::cout << "Union:        ";
    for (int v : result) std::cout << v << " ";
    std::cout << "\n";

    SetIntersection(a, b, result);
    std::cout << "Intersection: ";
    for (int v : result) std::cout << v << " ";
    std::cout << "\n";

    SetDifference(a, b, result);
    std::cout << "Difference:   ";
    for (int v : result) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
