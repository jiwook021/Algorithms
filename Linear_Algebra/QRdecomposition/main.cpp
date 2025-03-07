#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <mutex>

constexpr double EPSILON = 1e-10;

class QRDecomposer {
private:
    mutable std::mutex mutexLock; // For thread safety

    // Computes the dot product between two vectors.
    double computeDotProduct(const std::vector<double>& vectorA, const std::vector<double>& vectorB) const {
        if (vectorA.size() != vectorB.size()) {
            throw std::invalid_argument("Vectors must have the same length for dot product.");
        }
        double dotProductResult = 0.0;
        for (std::size_t index = 0; index < vectorA.size(); ++index) {
            dotProductResult += vectorA[index] * vectorB[index];
        }
        return dotProductResult;
    }

    // Computes the Euclidean norm (length) of a vector.
    double computeNorm(const std::vector<double>& vectorData) const {
        return std::sqrt(computeDotProduct(vectorData, vectorData));
    }

public:
    // Performs QR decomposition using the Gram–Schmidt process.
    // Input: inputMatrix (m x n matrix)
    // Output: pair of matrices (orthonormalMatrix Q, upperTriangularMatrix R) such that inputMatrix = Q * R.
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    decompose(const std::vector<std::vector<double>>& inputMatrix) const {
        std::lock_guard<std::mutex> lock(mutexLock); // Ensure thread-safety

        if (inputMatrix.empty() || inputMatrix[0].empty()) {
            throw std::invalid_argument("Input matrix cannot be empty.");
        }

        const std::size_t numberOfRows = inputMatrix.size();
        const std::size_t numberOfColumns = inputMatrix[0].size();

        // Initialize the orthonormal matrix (Q) and the upper triangular matrix (R)
        std::vector<std::vector<double>> orthonormalMatrix(numberOfRows, std::vector<double>(numberOfColumns, 0.0));
        std::vector<std::vector<double>> upperTriangularMatrix(numberOfColumns, std::vector<double>(numberOfColumns, 0.0));

        // Create a temporary storage for the intermediate orthogonalized columns.
        // Each column is stored as a vector of length 'numberOfRows'.
        std::vector<std::vector<double>> orthogonalColumns(numberOfColumns, std::vector<double>(numberOfRows, 0.0));

        // Copy columns from inputMatrix into orthogonalColumns.
        // This is a column-major copy: orthogonalColumns[col][row] = inputMatrix[row][col]
        for (std::size_t columnIndex = 0; columnIndex < numberOfColumns; ++columnIndex) {
            for (std::size_t rowIndex = 0; rowIndex < numberOfRows; ++rowIndex) {
                orthogonalColumns[columnIndex][rowIndex] = inputMatrix[rowIndex][columnIndex];
            }
        }

        // Apply Gram–Schmidt orthogonalization.
        for (std::size_t currentColumnIndex = 0; currentColumnIndex < numberOfColumns; ++currentColumnIndex) {

            for (std::size_t previousColumnIndex = 0; previousColumnIndex < currentColumnIndex; ++previousColumnIndex) {
                upperTriangularMatrix[previousColumnIndex][currentColumnIndex] =
                    computeDotProduct(orthogonalColumns[currentColumnIndex], orthonormalMatrix[previousColumnIndex]);
                for (std::size_t rowIndex = 0; rowIndex < numberOfRows; ++rowIndex) {
                    orthogonalColumns[currentColumnIndex][rowIndex] -=
                        upperTriangularMatrix[previousColumnIndex][currentColumnIndex] * orthonormalMatrix[previousColumnIndex][rowIndex];
                }
            }

            // Compute the norm of the current orthogonalized column.
            upperTriangularMatrix[currentColumnIndex][currentColumnIndex] = computeNorm(orthogonalColumns[currentColumnIndex]);
            if (upperTriangularMatrix[currentColumnIndex][currentColumnIndex] < EPSILON) {
                throw std::runtime_error("Matrix columns are linearly dependent.");
            }

            // Normalize the current column to form an orthonormal column.
            for (std::size_t rowIndex = 0; rowIndex < numberOfRows; ++rowIndex) {
                orthonormalMatrix[rowIndex][currentColumnIndex] =
                    orthogonalColumns[currentColumnIndex][rowIndex] / upperTriangularMatrix[currentColumnIndex][currentColumnIndex];
            }
        }

        return {orthonormalMatrix, upperTriangularMatrix};
    }
};

int main() {
    try {
        QRDecomposer qrDecomposer;

        // Define an example 3x3 matrix.
        std::vector<std::vector<double>> inputMatrix = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        // Perform the QR decomposition.
        auto [orthonormalMatrix, upperTriangularMatrix] = qrDecomposer.decompose(inputMatrix);

        // Display the orthonormal matrix Q.
        std::cout << "Orthonormal Matrix Q:\n";
        for (const auto& row : orthonormalMatrix) {
            for (double value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }

        // Display the upper triangular matrix R.
        std::cout << "\nUpper Triangular Matrix R:\n";
        for (const auto& row : upperTriangularMatrix) {
            for (double value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& exceptionMessage) {
        std::cerr << "Error: " << exceptionMessage.what() << "\n";
    }

    return 0;
}
//https://claude.ai/chat/64f25c82-9dcd-4e1e-9dc6-19950bb55732