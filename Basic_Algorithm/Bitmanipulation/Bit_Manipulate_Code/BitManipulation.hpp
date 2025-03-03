/**
 * @file BitManipulation.hpp
 * @brief Bit manipulation, number reversal, palindrome, power, and rectangle intersection utilities
 */

#pragma once

#include <cstdint>

namespace BitManipulation {

/**
 * @brief Reverse all 8 bits of a byte.
 */
char ReverseBit(char input);

/**
 * @brief Extract k bits starting from position s (1-based).
 */
int BitExtracted(int number, int k, int s);

/**
 * @brief Swap bits at positions p1 and p2 in n.
 */
unsigned int SwapBits(unsigned int n, int p1, int p2);

/**
 * @brief Reverse the decimal digits of an integer.
 */
int ReverseDigits(int num);

/**
 * @brief Check if num is a palindrome (reads same forwards and backwards).
 */
bool IsPalindrome(int num);

/**
 * @brief Compute a^x using simple iteration.
 */
double Power(double a, int x);

struct Rectangle {
    int X;
    int Y;
    int Width;
    int Height;
};

/**
 * @brief Compute intersection of two rectangles.
 * @return true if they intersect (result written to intersection)
 */
bool Intersect(const Rectangle& r1, const Rectangle& r2, Rectangle& intersection);

} // namespace BitManipulation
