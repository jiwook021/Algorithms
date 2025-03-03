/**
 * @file main.cpp
 * @brief Driver for Bitset
 */

#include "Bitset.hpp"
#include <iostream>

int main() {
    MyBitset<8> bitset;

    bitset.Set(1);
    bitset.Set(3);
    bitset.Set(7);
    bitset.Print();

    std::cout << "Test bit 1: " << bitset.Test(1) << '\n';
    std::cout << "Test bit 2: " << bitset.Test(2) << '\n';

    bitset.Flip(1);
    bitset.Print();

    bitset.Reset(3);
    bitset.Print();

    bitset.FlipAll();
    bitset.Print();

    std::cout << "Any? " << bitset.Any() << '\n';
    std::cout << "All? " << bitset.All() << '\n';
    std::cout << "None? " << bitset.None() << '\n';

    try {
        bitset.Test(8);
    } catch (const std::out_of_range& e) {
        std::cerr << "Exception caught: " << e.what() << '\n';
    }

    return 0;
}
