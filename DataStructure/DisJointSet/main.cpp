/**
 * @file main.cpp
 * @brief Driver for DisjointSet (Union-Find)
 */

#include "DisjointSet.hpp"
#include <iostream>

int main() {
    DisjointSet ds(10);

    ds.Union(1, 2);
    ds.Union(2, 3);
    ds.Union(3, 4);
    ds.Union(5, 6);
    ds.Union(6, 7);
    ds.Union(7, 8);

    std::cout << "1 and 5 connected? " << ds.Connected(1, 5) << "\n";

    ds.Union(1, 5);

    std::cout << "1 and 5 connected? " << ds.Connected(1, 5) << "\n";

    return 0;
}
