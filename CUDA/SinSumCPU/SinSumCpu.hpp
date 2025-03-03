#ifndef SIN_SUM_CPU_HPP
#define SIN_SUM_CPU_HPP

/**
 * @file SinSumCpu.hpp
 * @brief Header for CPU sine sum computation using Taylor series.
 *
 * Reference: "Programming in Parallel with CUDA" by Richard Ansorge (Example 1.1).
 */

/**
 * @brief Compute sin(x) using Taylor series expansion.
 * sin(x) = x - x^3/3! + x^5/5! - ...
 *
 * @param x     Input value.
 * @param terms Number of Taylor series terms.
 * @return Approximation of sin(x).
 */
inline float SinSum(float x, int terms) {
    float term = x;
    float sum  = term;
    float x2   = x * x;
    for (int n = 1; n < terms; n++) {
        term *= -x2 / (float)(2 * n * (2 * n + 1));
        sum += term;
    }
    return sum;
}

#endif // SIN_SUM_CPU_HPP
