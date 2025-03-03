/**
 * @file Attention.cpp
 * @brief Implementation of scaled dot-product attention.
 */
#include "Attention.hpp"

namespace Ml {

Matrix MatMul(const Matrix& a, const Matrix& b) {
    if (a.empty() || b.empty() || a[0].size() != b.size())
        throw std::invalid_argument("Matrix dimensions not compatible for multiplication");
    size_t n = a.size(), m = a[0].size(), p = b[0].size();
    Matrix r(n, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k)
            for (size_t j = 0; j < p; ++j)
                r[i][j] += a[i][k] * b[k][j];
    return r;
}

Matrix Transpose(const Matrix& m) {
    if (m.empty()) return {};
    size_t rows = m.size(), cols = m[0].size();
    Matrix r(cols, std::vector<double>(rows, 0.0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            r[j][i] = m[i][j];
    return r;
}

std::vector<double> Softmax(const std::vector<double>& input) {
    std::vector<double> out(input.size());
    double maxVal = *std::max_element(input.begin(), input.end());
    double sumExp = 0.0;
    for (double x : input) sumExp += std::exp(x - maxVal);
    for (size_t i = 0; i < input.size(); ++i)
        out[i] = std::exp(input[i] - maxVal) / sumExp;
    return out;
}

Matrix SoftmaxRowWise(const Matrix& m) {
    Matrix r = m;
    for (auto& row : r) row = Softmax(row);
    return r;
}

Matrix ScaledDotProductAttention::ComputeAttention(
    const Matrix& query, const Matrix& key, const Matrix& value) const {
    if (query.empty() || key.empty() || value.empty())
        throw std::invalid_argument("Input matrices cannot be empty");
    size_t dk = query[0].size();
    if (key.size() != value.size())
        throw std::invalid_argument("Key and Value row counts must match");
    Matrix scores = MatMul(query, Transpose(key));
    double scale = 1.0 / std::sqrt(static_cast<double>(dk));
    for (auto& row : scores) for (auto& s : row) s *= scale;
    return SoftmaxRowWise(scores);
}

} // namespace Ml
