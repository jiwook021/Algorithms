#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

// Define a small epsilon for floating-point comparisons.
constexpr double EPSILON = 1e-9;

// Computes the dot product of two vectors.
// Time Complexity: O(n) where n is the vector size.
// Throws std::invalid_argument if the vectors are not of equal dimension.
double dotProduct(const std::vector<double>& vectorA, const std::vector<double>& vectorB) {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("Vectors must have the same dimension for dot product.");
    }
    double product = 0.0;
    for (std::size_t i = 0; i < vectorA.size(); ++i) {
        product += vectorA[i] * vectorB[i];
    }
    return product;
}

// Computes the Euclidean norm (L2 norm) of a vector.
// Time Complexity: O(n)
double norm(const std::vector<double>& vectorA) {
    return std::sqrt(dotProduct(vectorA, vectorA));
}

// Multiplies a vector by a scalar.
// Time Complexity: O(n)
std::vector<double> scalarMultiply(const std::vector<double>& vectorA, double scalar) {
    std::vector<double> result(vectorA.size());
    for (std::size_t i = 0; i < vectorA.size(); ++i) {
        result[i] = vectorA[i] * scalar;
    }
    return result;
}

// Subtracts vectorB from vectorA element-wise.
// Time Complexity: O(n)
// Throws std::invalid_argument if vectors have different sizes.
std::vector<double> subtractVectors(const std::vector<double>& vectorA, const std::vector<double>& vectorB) {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("Vectors must have the same dimension for subtraction.");
    }
    std::vector<double> result(vectorA.size());
    for (std::size_t i = 0; i < vectorA.size(); ++i) {
        result[i] = vectorA[i] - vectorB[i];
    }
    return result;
}

// Performs the Gram–Schmidt process to generate an orthonormal basis from a set of input vectors.
// Time Complexity: O(n^2 * m) where n is the number of input vectors and m is the dimension of each vector.
// Memory Complexity: O(n * m)
// Throws std::invalid_argument if an input vector is empty.
std::vector<std::vector<double>> gramSchmidt(const std::vector<std::vector<double>>& inputVectors) {
    std::vector<std::vector<double>> orthonormalBasis;

    for (const auto& vector : inputVectors) {
        if (vector.empty()) {
            throw std::invalid_argument("Input vector is empty.");
        }
        // Start with the original vector.
        std::vector<double> orthogonalVector = vector;
        
        // Remove components in the directions of the already computed basis vectors.
        for (const auto& basisVector : orthonormalBasis) {
            double projectionCoefficient = dotProduct(orthogonalVector, basisVector);
            auto projection = scalarMultiply(basisVector, projectionCoefficient);
            orthogonalVector = subtractVectors(orthogonalVector, projection);
        }
        
        // Compute the norm of the orthogonalized vector.
        double vectorNorm = norm(orthogonalVector);
        // If the vector is nearly zero, it is linearly dependent on the others; skip it.
        if (vectorNorm < EPSILON) {
            continue;
        }
        // Normalize the vector to get the unit vector.
        auto normalizedVector = scalarMultiply(orthogonalVector, 1.0 / vectorNorm);
        orthonormalBasis.push_back(normalizedVector);
    }
    return orthonormalBasis;
}

// Multiplies two matrices A and B.
// Matrices are represented as vectors of vectors. Each inner vector is a row.
// Time Complexity: O(A_rows * A_cols * B_cols)
// Throws std::invalid_argument if dimensions are incompatible or row sizes are inconsistent.
std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& matrixA,
                                                    const std::vector<std::vector<double>>& matrixB) {
    if (matrixA.empty() || matrixB.empty()) {
        throw std::invalid_argument("Empty matrix provided.");
    }
    std::size_t A_rows = matrixA.size();
    std::size_t A_cols = matrixA[0].size();
    std::size_t B_rows = matrixB.size();
    std::size_t B_cols = matrixB[0].size();

    if (A_cols != B_rows) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    // Initialize the result matrix with zeros.
    std::vector<std::vector<double>> result(A_rows, std::vector<double>(B_cols, 0.0));
    for (std::size_t i = 0; i < A_rows; ++i) {
        if (matrixA[i].size() != A_cols) {
            throw std::invalid_argument("Inconsistent row sizes in the first matrix.");
        }
        for (std::size_t j = 0; j < B_cols; ++j) {
            for (std::size_t k = 0; k < A_cols; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return result;
}

// Computes the transpose of a matrix.
// Time Complexity: O(rows * cols)
// Throws std::invalid_argument if the matrix is empty or has inconsistent row sizes.
std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Empty matrix provided.");
    }
    std::size_t rows = matrix.size();
    std::size_t cols = matrix[0].size();
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows, 0.0));
    
    for (std::size_t i = 0; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw std::invalid_argument("Inconsistent row sizes in the matrix.");
        }
        for (std::size_t j = 0; j < cols; ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }
    return transposedMatrix;
}

// Checks whether a given square matrix is orthogonal.
// A matrix is orthogonal if A * A^T equals the identity matrix.
// Time Complexity: O(n^3) for an n x n matrix.
// Memory Complexity: O(n^2)
// Throws std::invalid_argument if the matrix is not square or is empty.
bool isOrthogonalMatrix(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Empty matrix provided.");
    }
    std::size_t n = matrix.size();
    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("Matrix is not square.");
        }
    }
    // Compute A * A^T.
    auto matrixTranspose = transposeMatrix(matrix);
    auto productMatrix = multiplyMatrices(matrix, matrixTranspose);
    
    // Check if productMatrix is an identity matrix.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double expectedValue = (i == j) ? 1.0 : 0.0;
            if (std::abs(productMatrix[i][j] - expectedValue) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    try {
        // Demonstration: Orthonormalization using Gram–Schmidt process.
        std::cout << "Gram–Schmidt Orthonormalization Example:" << std::endl;
        std::vector<std::vector<double>> inputVectors = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        auto orthonormalBasis = gramSchmidt(inputVectors);

        std::cout << "Computed Orthonormal Basis:" << std::endl;
        for (const auto& vec : orthonormalBasis) {
            for (double value : vec) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }

        // Demonstration: Check if a given matrix is orthogonal.
        std::cout << "\nOrthogonal Matrix Check Example:" << std::endl;
        std::vector<std::vector<double>> testOrthogonalMatrix = {
            {1.0, 0.0, 0.0},
            {0.0, 0.0, -1.0},
            {0.0, 1.0, 0.0}
        };

        if (isOrthogonalMatrix(testOrthogonalMatrix)) {
            std::cout << "The matrix is orthogonal." << std::endl;
        } else {
            std::cout << "The matrix is NOT orthogonal." << std::endl;
        }

    } catch (const std::exception& ex) {
        // Catch and report any errors during computation.
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}

