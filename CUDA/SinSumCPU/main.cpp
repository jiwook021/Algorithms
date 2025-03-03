/**
 * @file main.cpp
 * @brief Driver for CPU sine sum computation using Taylor series.
 *
 * Numerically integrates sin(x) from 0 to pi using the trapezoidal rule.
 * Expected result: integral of sin(x) from 0 to pi = 2.0.
 *
 * Usage: ./main [steps] [terms]
 *
 * Reference: "Programming in Parallel with CUDA" by Richard Ansorge (Example 1.1).
 */

#include "SinSumCpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;

    double pi       = 3.14159265358979323;
    double stepSize = pi / (steps - 1);

    auto start = std::chrono::high_resolution_clock::now();

    double cpuSum = 0.0;
    for (int step = 0; step < steps; step++) {
        float x = (float)(stepSize * step);
        cpuSum += SinSum(x, terms);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Trapezoidal rule correction for end points
    cpuSum -= 0.5f * (SinSum(0.0f, terms) + SinSum((float)pi, terms));
    cpuSum *= stepSize;

    printf("cpu sum = %.10f, steps %d terms %d time %.3f ms\n",
           cpuSum, steps, terms, elapsed.count());

    return 0;
}
