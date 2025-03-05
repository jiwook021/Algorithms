#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

//------------------------------------------------------------------------------
// Function: isSquareMatrix
// Description:
//   Checks if the given matrix (vector of vectors) is square.
// Parameters:
//   - matrix: A 2D vector representing the matrix.
// Returns:
//   true if the matrix is square, false otherwise.
//------------------------------------------------------------------------------
bool isSquareMatrix(const std::vector<std::vector<double>>& matrix) {
    size_t numRows = matrix.size();
    if (numRows == 0) return false;
    for (const auto& row : matrix) {
        if (row.size() != numRows) {
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Function: createIdentityMatrix
// Description:
//   Creates an identity matrix of the given size.
// Parameters:
//   - size: The size (number of rows/columns) of the identity matrix.
// Returns:
//   A 2D vector representing the identity matrix.
//------------------------------------------------------------------------------
std::vector<std::vector<double>> createIdentityMatrix(size_t size) {
    std::vector<std::vector<double>> identity(size, std::vector<double>(size, 0.0));
    for (size_t i = 0; i < size; ++i) {
        identity[i][i] = 1.0;
    }
    return identity;
}

//------------------------------------------------------------------------------
// Function: jacobiEigenDecomposition
// Description:
//   Computes the eigendecomposition of a symmetric matrix using the Jacobi method.
//   It returns a pair containing a vector of eigenvalues (from the diagonal of the
//   transformed matrix) and a matrix of corresponding eigenvectors (accumulated rotations).
// Parameters:
//   - matrix: A 2D vector representing a symmetric square matrix.
//   - tolerance: The convergence tolerance for off-diagonal elements.
//   - maxIterations: Maximum number of iterations allowed.
// Returns:
//   A pair with:
//     - first: A vector of eigenvalues.
//     - second: A 2D vector representing the eigenvectors (each column is an eigenvector).
// Throws:
//   - std::invalid_argument if the input matrix is not square.
// Time Complexity: O(n^3 * iterations) where n is the matrix dimension.
// Space Complexity: O(n^2)
//------------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<std::vector<double>>> 
jacobiEigenDecomposition(std::vector<std::vector<double>> matrix, 
                         double tolerance = 1e-10, 
                         size_t maxIterations = 100)
{
    if (!isSquareMatrix(matrix)) {
        throw std::invalid_argument("Input matrix must be square.");
    }

    const size_t n = matrix.size();
    // Initialize eigenvector matrix as the identity matrix.
    auto eigenvectors = createIdentityMatrix(n);

    for (size_t iter = 0; iter < maxIterations; ++iter) {
        // Find the largest off-diagonal element in absolute value.
        size_t p = 0, q = 1;
        double maxOffDiagonal = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double absValue = std::fabs(matrix[i][j]);
                if (absValue > maxOffDiagonal) {
                    maxOffDiagonal = absValue;
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence: if the maximum off-diagonal element is below the tolerance.
        if (maxOffDiagonal < tolerance) {
            break;
        }

        // Compute the Jacobi rotation angle.
        double theta = 0.0;
        if (std::fabs(matrix[q][q] - matrix[p][p]) < tolerance) {
            theta = (matrix[p][q] > 0 ? M_PI / 4.0 : -M_PI / 4.0);
        } else {
            theta = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        }
        double cosTheta = std::cos(theta);
        double sinTheta = std::sin(theta);

        // Save the current diagonal elements.
        double a_pp = matrix[p][p];
        double a_qq = matrix[q][q];
        double a_pq = matrix[p][q];

        // Update the diagonal elements.
        matrix[p][p] = cosTheta * cosTheta * a_pp - 2.0 * sinTheta * cosTheta * a_pq + sinTheta * sinTheta * a_qq;
        matrix[q][q] = sinTheta * sinTheta * a_pp + 2.0 * sinTheta * cosTheta * a_pq + cosTheta * cosTheta * a_qq;
        // Set the off-diagonal elements at (p, q) and (q, p) to zero.
        matrix[p][q] = matrix[q][p] = 0.0;

        // Update the remaining elements.
        for (size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double a_ip = matrix[i][p];
                double a_iq = matrix[i][q];
                // Apply rotation.
                matrix[i][p] = matrix[p][i] = cosTheta * a_ip - sinTheta * a_iq;
                matrix[i][q] = matrix[q][i] = sinTheta * a_ip + cosTheta * a_iq;
            }
        }

        // Update the eigenvectors matrix.
        for (size_t i = 0; i < n; ++i) {
            double v_ip = eigenvectors[i][p];
            double v_iq = eigenvectors[i][q];
            eigenvectors[i][p] = cosTheta * v_ip - sinTheta * v_iq;
            eigenvectors[i][q] = sinTheta * v_ip + cosTheta * v_iq;
        }
    }

    // The eigenvalues are now on the diagonal of the matrix.
    std::vector<double> eigenvalues(n);
    for (size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }

    return { eigenvalues, eigenvectors };
}

//------------------------------------------------------------------------------
// Main function demonstrating the usage of the jacobiEigenDecomposition function.
//------------------------------------------------------------------------------
int main() {
    try {
        // Define a sample symmetric 3x3 matrix.
        std::vector<std::vector<double>> sampleMatrix = {
            { 4.0, -2.0,  2.0 },
            {-2.0,  2.0, -4.0 },
            { 2.0, -4.0, 11.0 }
        };

        // Perform the eigendecomposition.
        auto [eigenvalues, eigenvectors] = jacobiEigenDecomposition(sampleMatrix);

        // Output the computed eigenvalues.
        std::cout << "Eigenvalues:\n";
        for (const auto& value : eigenvalues) {
            std::cout << value << "\n";
        }
        std::cout << "\nEigenvectors (each column is an eigenvector):\n";
        // Print eigenvectors in a matrix format.
        for (size_t i = 0; i < eigenvectors.size(); ++i) {
            for (size_t j = 0; j < eigenvectors.size(); ++j) {
                std::cout << eigenvectors[i][j] << "\t";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "Error during eigendecomposition: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
