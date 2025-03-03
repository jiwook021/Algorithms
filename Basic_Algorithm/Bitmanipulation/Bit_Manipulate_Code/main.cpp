/**
 * @file main.cpp
 * @brief Driver for BitManipulation
 */

#include "BitManipulation.hpp"
#include <cstdio>

using namespace BitManipulation;

int main() {
    printf("ReverseBit(12): %d\n", ReverseBit(12));
    printf("BitExtracted(108, 4, 3): %d\n", BitExtracted(108, 4, 3));
    printf("SwapBits(28, 0, 3): %u\n", SwapBits(28, 0, 3));
    printf("ReverseDigits(123): %d\n", ReverseDigits(123));
    printf("IsPalindrome(12321): %d\n", IsPalindrome(12321));
    printf("Power(2, 10): %.0f\n", Power(2, 10));

    Rectangle r1 = {0, 0, 4, 4};
    Rectangle r2 = {2, 2, 4, 4};
    Rectangle intersection;
    if (Intersect(r1, r2, intersection)) {
        printf("Intersection: x=%d, y=%d, w=%d, h=%d\n",
               intersection.X, intersection.Y,
               intersection.Width, intersection.Height);
    }

    return 0;
}
