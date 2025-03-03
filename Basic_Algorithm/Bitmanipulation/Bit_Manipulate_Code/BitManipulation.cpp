/**
 * @file BitManipulation.cpp
 * @brief Implementation of bit manipulation utilities
 */

#include "BitManipulation.hpp"
#include <algorithm>

namespace BitManipulation {

char ReverseBit(char input) {
    char result = 0;
    for (int i = 0; i < 8; ++i) {
        result <<= 1;
        result |= (input & 1);
        input >>= 1;
    }
    return result;
}

int BitExtracted(int number, int k, int s) {
    return ((1 << k) - 1) & (number >> (s - 1));
}

unsigned int SwapBits(unsigned int n, int p1, int p2) {
    unsigned int bit1 = (n >> p1) & 1;
    unsigned int bit2 = (n >> p2) & 1;
    unsigned int x = bit1 ^ bit2;
    x = (x << p1) | (x << p2);
    return n ^ x;
}

int ReverseDigits(int num) {
    int result = 0;
    while (num != 0) {
        result = result * 10 + num % 10;
        num /= 10;
    }
    return result;
}

bool IsPalindrome(int num) {
    if (num < 0) return false;
    return num == ReverseDigits(num);
}

double Power(double a, int x) {
    if (x == 0) return 1.0;
    if (x < 0) return 1.0 / Power(a, -x);
    double half = Power(a, x / 2);
    return (x % 2 == 0) ? half * half : half * half * a;
}

static int Min(int a, int b) { return (a < b) ? a : b; }
static int Max(int a, int b) { return (a > b) ? a : b; }

bool Intersect(const Rectangle& r1, const Rectangle& r2, Rectangle& intersection) {
    int xOverlap = Min(r1.X + r1.Width, r2.X + r2.Width) - Max(r1.X, r2.X);
    int yOverlap = Min(r1.Y + r1.Height, r2.Y + r2.Height) - Max(r1.Y, r2.Y);

    if (xOverlap > 0 && yOverlap > 0) {
        intersection.X = Max(r1.X, r2.X);
        intersection.Y = Max(r1.Y, r2.Y);
        intersection.Width = xOverlap;
        intersection.Height = yOverlap;
        return true;
    }
    return false;
}

} // namespace BitManipulation
