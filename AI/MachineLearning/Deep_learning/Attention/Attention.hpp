/**
 * @file Attention.hpp
 * @brief Scaled dot-product attention mechanism
 * @details Implements the core transformer attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
 *          Includes matrix multiply, transpose, and row-wise softmax utilities.
 */
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace Ml {

using Matrix = std::vector<std::vector<double>>;

Matrix MatMul(const Matrix& a, const Matrix& b);
Matrix Transpose(const Matrix& m);
std::vector<double> Softmax(const std::vector<double>& input);
Matrix SoftmaxRowWise(const Matrix& m);

/**
 * @brief Scaled dot-product attention.
 */
class ScaledDotProductAttention {
public:
    /**
     * @brief Compute attention weights from Q, K, V matrices.
     * @return Attention weight matrix (numQueries x numKeys).
     */
    Matrix ComputeAttention(const Matrix& query, const Matrix& key, const Matrix& value) const;
};

} // namespace Ml
