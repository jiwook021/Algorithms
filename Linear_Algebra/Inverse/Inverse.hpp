/**
 * @file Inverse.hpp
 * @brief Dense matrix with Gauss-Jordan inversion (declarations).
 *
 * @details
 * A dense matrix stores numbers in a grid (rows x cols).
 * The Inverse() method finds A^{-1} such that A * A^{-1} = I
 * using Gauss-Jordan elimination on the augmented matrix [A | I].
 *
 * Complexity:
 *   Inverse : O(n^3) time, O(n^2) space
 *   At      : O(1)
 */

#ifndef INVERSE_HPP
#define INVERSE_HPP

#include <cstddef>
#include <initializer_list>
#include <vector>

/**
 * @class Matrix
 * @brief Dense matrix with Gauss-Jordan inversion.
 *
 * @tparam Scalar Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class Matrix {
public:
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    /** @brief Create a zero-initialized rows x cols matrix. */
    Matrix(std::size_t rows, std::size_t cols);

    /** @brief Create from a braced initializer list of rows. */
    Matrix(std::initializer_list<std::initializer_list<Scalar>> init);

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    [[nodiscard]] std::size_t Rows() const;
    [[nodiscard]] std::size_t Cols() const;

    Scalar& At(std::size_t row, std::size_t col);
    const Scalar& At(std::size_t row, std::size_t col) const;

    // ---------------------------------------------------------------
    // Core operation
    // ---------------------------------------------------------------

    /**
     * @brief Compute the inverse via Gauss-Jordan elimination.
     * @return A new Matrix that is A^{-1}.
     * @throws std::runtime_error if non-square or singular.
     */
    [[nodiscard]] Matrix Inverse() const;

    /** @brief Print the matrix to stdout. */
    void Print() const;

private:
    static constexpr Scalar kPivotTolerance = static_cast<Scalar>(1e-12);

    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::vector<Scalar>> data_;

    // Gauss-Jordan helpers (make the algorithm readable)

    /** Find the row (from start_row..n-1) with the largest |value| in column col. */
    std::size_t FindBestPivotRow(
        const std::vector<std::vector<Scalar>>& aug,
        std::size_t col, std::size_t start_row) const;

    /** Divide every element in row by its diagonal entry, making the pivot = 1. */
    void ScaleRowToMakePivotOne(
        std::vector<std::vector<Scalar>>& aug,
        std::size_t row, std::size_t width) const;

    /** Subtract multiples of the pivot row from every other row to zero out the column. */
    void ZeroOutColumn(
        std::vector<std::vector<Scalar>>& aug,
        std::size_t pivot_row, std::size_t n) const;
};

#endif // INVERSE_HPP
