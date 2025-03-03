/**
 * @file main.cpp
 * @brief Driver for Attention -- scaled dot-product attention demo.
 */
#include "Attention.hpp"
#include <iostream>

int main() {
    using namespace Ml;
    Matrix q = {{1,0,1,0}, {0,1,0,1}};
    Matrix k = {{1,0,1,0}, {0,1,0,1}, {1,1,0,0}};
    Matrix v = {{1,2,3}, {4,5,6}, {7,8,9}};

    ScaledDotProductAttention attn;
    auto weights = attn.ComputeAttention(q, k, v);

    std::cout << "Attention weights:\n";
    for (const auto& row : weights) {
        for (double x : row) std::cout << x << " ";
        std::cout << "\n";
    }
    return 0;
}
