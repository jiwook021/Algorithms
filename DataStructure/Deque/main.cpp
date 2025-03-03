/**
 * @file main.cpp
 * @brief Driver for Deque
 */

#include "Deque.hpp"
#include <iostream>

int main() {
    Deque<int> dq{1, 2, 3, 4, 5};

    std::cout << "Front: " << dq.Front() << ", Back: " << dq.Back() << "\n";
    std::cout << "Size: " << dq.Size() << "\n";

    dq.PushFront(0);
    dq.PushBack(6);

    std::cout << "After push front/back: ";
    for (int v : dq) std::cout << v << " ";
    std::cout << "\n";

    dq.PopFront();
    dq.PopBack();

    std::cout << "After pop front/back: ";
    for (int v : dq) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
