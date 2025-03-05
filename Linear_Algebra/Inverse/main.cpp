#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <utility> // for std::swap

// Matrix class to encapsulate matrix data and operations.
class Matrix {
public:
    // Constructor that accepts a 2D vector representing the matrix elements.
    // Throws invalid_argument if the matrix is empty or if rows have inconsistent sizes.
    Matrix(const std::vector<std::vector<double>>& elements);

    // Returns the number of rows in the matrix.
    size_t rows() const;

    // Returns the number of columns in the matrix.
    size_t cols() const;

    // Computes and returns the inverse of the current square matrix.
    // Throws runtime_error if the matrix is not square or is singular (non-invertible).
    Matrix inverse() const;

    // Prints the matrix to standard output.
    void print() const;

private:
    // Stores the matrix elements.
    std::vector<std::vector<double>> data_;
};

// Implementation of the Matrix constructor.
Matrix::Matrix(const std::vector<std::vector<double>>& elements) {
    if (elements.empty() || elements[0].empty()) {
        throw std::invalid_argument("Matrix cannot be empty.");
    }
    size_t columnSize = elements[0].size();
    for (const auto& row : elements) {
        if (row.size() != columnSize) {
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
    }
    data_ = elements;
}

// Returns the number of rows.
size_t Matrix::rows() const {
    return data_.size();
}

// Returns the number of columns.
size_t Matrix::cols() const {
    return data_[0].size();
}

// Prints the matrix with formatted output.
void Matrix::print() const {
    for (const auto& row : data_) {
        for (double value : row) {
            std::cout << std::setw(10) << value << " ";
        }
        std::cout << "\n";
    }
}

// Computes the inverse of a square matrix using Gauss-Jordan elimination.
// Time Complexity: O(n^3), where n is the dimension of the matrix.
// Memory Complexity: O(n^2) extra space is used for the augmented matrix.
Matrix Matrix::inverse() const {
    // Ensure the matrix is square.
    if (rows() != cols()) {
        throw std::runtime_error("Only square matrices can be inverted.");
    }
    size_t n = rows();

    // Create an augmented matrix [A | I] of dimensions n x 2n.
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));

    // Initialize the augmented matrix with the original matrix and the identity matrix.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i][j] = data_[i][j];
        }
        augmented[i][n + i] = 1.0; // Set the identity matrix.
    }

    // Perform Gauss-Jordan elimination.
    for (size_t i = 0; i < n; i++) {
        // Find the pivot row by selecting the row with the maximum absolute value in column i.
        size_t pivotRow = i;
        double maxElement = std::abs(augmented[i][i]);
        for (size_t k = i + 1; k < n; k++) {
            double currentElement = std::abs(augmented[k][i]);
            if (currentElement > maxElement) {
                maxElement = currentElement;
                pivotRow = k;
            }
        }

        // If the pivot element is effectively zero, the matrix is singular.
        if (std::abs(augmented[pivotRow][i]) < 1e-12) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        // Swap the current row with the pivot row if necessary.
        if (pivotRow != i) {
            std::swap(augmented[i], augmented[pivotRow]);
        }

        // Normalize the pivot row by dividing all elements by the pivot element.
        double pivotValue = augmented[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivotValue;
        }

        // Eliminate all other entries in column i.
        for (size_t row = 0; row < n; row++) {
            if (row != i) {
                double factor = augmented[row][i];
                for (size_t col = 0; col < 2 * n; col++) {
                    augmented[row][col] -= factor * augmented[i][col];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix.
    std::vector<std::vector<double>> inverseData(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverseData[i][j] = augmented[i][n + j];
        }
    }

    return Matrix(inverseData);
}

// Main function to demonstrate matrix inversion.
int main() {
    try {
        // Example matrix (2x2) for inversion.
        std::vector<std::vector<double>> matrixElements = {
            {4, 7},
            {2, 6}
        };

        Matrix matrix(matrixElements);

        std::cout << "Original Matrix:\n";
        matrix.print();

        // Compute the inverse of the matrix.
        Matrix inverseMatrix = matrix.inverse();

        std::cout << "\nInverse Matrix:\n";
        inverseMatrix.print();
    }
    catch (const std::exception& ex) {
        // Handle errors such as non-square or singular matrices.
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
