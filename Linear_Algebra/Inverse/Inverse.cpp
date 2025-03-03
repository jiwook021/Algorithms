/**
 * @file Inverse.cpp
 * @brief Implementation of Matrix methods.
 */

#include "Inverse.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Scalar>
Matrix<Scalar>::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows, std::vector<Scalar>(cols, Scalar(0))) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero.");
    }
}

template <typename Scalar>
Matrix<Scalar>::Matrix(
        std::initializer_list<std::initializer_list<Scalar>> init) {
    if (init.size() == 0 || init.begin()->size() == 0) {
        throw std::invalid_argument("Matrix cannot be empty.");
    }
    rows_ = init.size();
    cols_ = init.begin()->size();
    data_.reserve(rows_);
    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument(
                "All rows must have the same number of columns.");
        }
        data_.emplace_back(row);
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <typename Scalar>
std::size_t Matrix<Scalar>::Rows() const { return rows_; }

template <typename Scalar>
std::size_t Matrix<Scalar>::Cols() const { return cols_; }

template <typename Scalar>
Scalar& Matrix<Scalar>::At(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range.");
    }
    return data_[row][col];
}

template <typename Scalar>
const Scalar& Matrix<Scalar>::At(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range.");
    }
    return data_[row][col];
}

template <typename Scalar>
void Matrix<Scalar>::Print() const {
    for (const auto& row : data_) {
        for (const Scalar& val : row) {
            std::cout << std::setw(12) << val << " ";
        }
        std::cout << "\n";
    }
}

// ---------------------------------------------------------------------------
// Gauss-Jordan helpers
// ---------------------------------------------------------------------------

template <typename Scalar>
std::size_t Matrix<Scalar>::FindBestPivotRow(
        const std::vector<std::vector<Scalar>>& aug,
        std::size_t col, std::size_t start_row) const {
    //
    // Scan rows from start_row to the bottom.
    // Pick the row whose entry in this column has the
    // largest absolute value.  Using the biggest value
    // as the pivot reduces floating-point rounding errors.
    //
    std::size_t best = start_row;
    Scalar best_val = std::abs(aug[start_row][col]);

    for (std::size_t row = start_row + 1; row < rows_; ++row) {
        Scalar val = std::abs(aug[row][col]);
        if (val > best_val) {
            best_val = val;
            best = row;
        }
    }
    return best;
}

template <typename Scalar>
void Matrix<Scalar>::ScaleRowToMakePivotOne(
        std::vector<std::vector<Scalar>>& aug,
        std::size_t row, std::size_t width) const {
    //
    // Divide every element in this row by the diagonal entry.
    // After this, the diagonal entry becomes exactly 1.
    //
    //   Before: [ ... 3 ...  6 ... ]  (pivot = 3)
    //   After:  [ ... 1 ...  2 ... ]  (divided by 3)
    //
    Scalar pivot = aug[row][row];
    for (std::size_t j = 0; j < width; ++j) {
        aug[row][j] /= pivot;
    }
}

template <typename Scalar>
void Matrix<Scalar>::ZeroOutColumn(
        std::vector<std::vector<Scalar>>& aug,
        std::size_t pivot_row, std::size_t n) const {
    //
    // For every row OTHER than the pivot row:
    //   subtract (that row's entry in this column) * (pivot row)
    //
    // This zeros out the entire column except for the pivot's 1.
    //
    //   Before:  pivot row = [ 0 1 0 | ... ]
    //            other row = [ 0 5 3 | ... ]
    //   factor = 5
    //   After:   other row = [ 0 0 3-5*0 | ... ] = [ 0 0 3 | ... ]
    //
    for (std::size_t row = 0; row < n; ++row) {
        if (row == pivot_row) continue;

        Scalar factor = aug[row][pivot_row];
        for (std::size_t col = 0; col < 2 * n; ++col) {
            aug[row][col] -= factor * aug[pivot_row][col];
        }
    }
}

// ---------------------------------------------------------------------------
// Gauss-Jordan elimination  (the main algorithm)
// ---------------------------------------------------------------------------

template <typename Scalar>
auto Matrix<Scalar>::Inverse() const -> Matrix {
    if (rows_ != cols_) {
        throw std::runtime_error("Only square matrices can be inverted.");
    }
    const std::size_t n = rows_;

    // ------------------------------------------------------------------
    // Step 0: Build the augmented matrix [A | I]
    //
    //   Left half = copy of A.   Right half = identity matrix.
    //   Example for a 2x2:
    //     [ a b | 1 0 ]
    //     [ c d | 0 1 ]
    //
    //   Goal: turn the left half into I using row operations.
    //   The same operations will turn the right half into A^{-1}.
    // ------------------------------------------------------------------

    std::vector<std::vector<Scalar>> aug(
        n, std::vector<Scalar>(2 * n, Scalar(0)));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            aug[i][j] = data_[i][j];     // copy A into left half
        }
        aug[i][n + i] = Scalar(1);       // identity in right half
    }

    // ------------------------------------------------------------------
    // Process each column to turn the left half into the identity.
    // After all columns are done, the right half is A^{-1}.
    // ------------------------------------------------------------------

    for (std::size_t col = 0; col < n; ++col) {

        // 1. Find the best pivot row (largest absolute value in this column).
        //    This is called "partial pivoting" and improves numerical accuracy.
        std::size_t best_row = FindBestPivotRow(aug, col, col);

        // 2. If the best pivot is effectively zero, the matrix is singular
        //    (no inverse exists).
        if (std::abs(aug[best_row][col]) < kPivotTolerance) {
            throw std::runtime_error("Matrix is singular");
        }

        // 3. Swap the best row into position (move it to the diagonal).
        if (best_row != col) {
            std::swap(aug[col], aug[best_row]);
        }

        // 4. Scale this row so the diagonal entry becomes 1.
        ScaleRowToMakePivotOne(aug, col, 2 * n);

        // 5. Subtract multiples of this row from every other row
        //    to make their entries in this column become 0.
        ZeroOutColumn(aug, col, n);
    }

    // ------------------------------------------------------------------
    // Extract the right half — that's the inverse.
    // ------------------------------------------------------------------

    Matrix<Scalar> result(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.At(i, j) = aug[i][n + j];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long double>;
