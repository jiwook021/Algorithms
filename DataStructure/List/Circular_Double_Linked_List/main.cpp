/**
 * @file main.cpp
 * @brief Driver for CircularList (sentinel-based circular doubly-linked list)
 */

#include "CircularList.hpp"
#include <iostream>

int main() {
    CircularList<int> cl{1, 2, 3, 4, 5};

    std::cout << "List: ";
    for (int v : cl) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Front: " << cl.Front() << ", Back: " << cl.Back() << "\n";
    std::cout << "Size: " << cl.Size() << "\n";

    cl.push_front(0);
    cl.push_back(6);

    std::cout << "After push: ";
    for (int v : cl) std::cout << v << " ";
    std::cout << "\n";

    cl.remove(3);
    std::cout << "After remove(3): ";
    for (int v : cl) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
