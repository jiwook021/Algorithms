#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

//------------------------------------------------------------------------------
// Helper function: dotProduct
// Computes the dot product of two vectors.
double dotProduct(const Vector& a, const Vector& b) {
    if(a.size() != b.size())
        throw std::invalid_argument("Vector sizes do not match for dot product.");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

//------------------------------------------------------------------------------
// Helper function: vectorNorm
// Computes the Euclidean norm (L2 norm) of a vector.
double vectorNorm(const Vector& a) {
    return std::sqrt(dotProduct(a, a));
}

//------------------------------------------------------------------------------
// Helper function: getColumn
// Extracts the j-th column from matrix A.
Vector getColumn(const Matrix &A, size_t j) {
    Vector col;
    for (const auto &row : A) {
        if(j >= row.size())
            throw std::invalid_argument("Column index out of bounds in getColumn.");
        col.push_back(row[j]);
    }
    return col;
}

//------------------------------------------------------------------------------
// Helper function: createZeroMatrix
// Creates an n x n matrix filled with zeros.
Matrix createZeroMatrix(size_t n) {
    return Matrix(n, Vector(n, 0.0));
}

//------------------------------------------------------------------------------
// Helper function: identityMatrix
// Creates an identity matrix of size n.
Matrix identityMatrix(size_t n) {
    Matrix I = createZeroMatrix(n);
    for (size_t i = 0; i < n; ++i)
        I[i][i] = 1.0;
    return I;
}

//------------------------------------------------------------------------------
// Helper function: multiplyMatrices
// Performs matrix multiplication: C = A * B.
Matrix multiplyMatrices(const Matrix &A, const Matrix &B) {
    if (A.empty() || B.empty() || A[0].size() != B.size())
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
    size_t m = A.size();
    size_t n = A[0].size();  // also equals B.size()
    size_t p = B[0].size();
    Matrix C(m, Vector(p, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j)
            for (size_t k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

//------------------------------------------------------------------------------
// Helper function: setColumn
// Sets the j-th column of matrix Q to vector v.
void setColumn(Matrix &Q, size_t j, const Vector &v) {
    if(Q.size() != v.size())
        throw std::invalid_argument("Mismatched sizes in setColumn.");
    for (size_t i = 0; i < Q.size(); ++i)
        Q[i][j] = v[i];
}

//------------------------------------------------------------------------------
// Function: qrDecomposition
// Performs a QR decomposition of a square matrix A using the classical Gram–Schmidt process.
// Returns a pair (Q, R) such that A = Q * R, where Q is orthonormal and R is upper triangular.
std::pair<Matrix, Matrix> qrDecomposition(const Matrix &A) {
    size_t n = A.size();
    // Ensure A is square.
    for (const auto &row : A)
        if (row.size() != n)
            throw std::invalid_argument("Matrix must be square for QR decomposition.");

    Matrix Q = createZeroMatrix(n);
    Matrix R = createZeroMatrix(n);

    // Process each column of A.
    for (size_t j = 0; j < n; ++j) {
        // Start with the j-th column of A.
        Vector v = getColumn(A, j);
        // Subtract the projection on each previous Q column.
        for (size_t i = 0; i < j; ++i) {
            Vector q_i = getColumn(Q, i);
            R[i][j] = dotProduct(q_i, getColumn(A, j));
            for (size_t k = 0; k < n; ++k)
                v[k] -= R[i][j] * q_i[k];
        }
        R[j][j] = vectorNorm(v);
        if (std::fabs(R[j][j]) < std::numeric_limits<double>::epsilon())
            throw std::runtime_error("Matrix has linearly dependent columns (QR decomposition failed).");
        // Normalize v to obtain the j-th column of Q.
        for (size_t k = 0; k < n; ++k)
            Q[k][j] = v[k] / R[j][j];
    }
    return {Q, R};
}

//------------------------------------------------------------------------------
// Helper function: offDiagonalFrobeniusNorm
// Computes the Frobenius norm of the off-diagonal elements of a square matrix.
double offDiagonalFrobeniusNorm(const Matrix &A) {
    size_t n = A.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() != n)
            throw std::invalid_argument("Matrix is not square in offDiagonalFrobeniusNorm.");
        for (size_t j = 0; j < n; ++j) {
            if (i != j)
                sum += A[i][j] * A[i][j];
        }
    }
    return std::sqrt(sum);
}

//------------------------------------------------------------------------------
// Function: qrEigenDecomposition
// Applies the QR algorithm to compute an approximate eigendecomposition of a general square matrix.
// It returns a pair containing a vector of eigenvalues (from the diagonal of the converged matrix)
// and a matrix of corresponding eigenvectors (accumulated Q matrices).
//
// Note: For non-symmetric matrices, the QR algorithm may converge to a Schur form where
// the eigenvalues are on the diagonal (or in 2x2 blocks for complex pairs). This implementation
// approximates the eigenvalues as the diagonal entries and the eigenvectors as the accumulated Q.
// The algorithm may not capture complex eigenpairs and does not use shifts.
std::pair<Vector, Matrix> qrEigenDecomposition(Matrix A, double tolerance = 1e-10, size_t maxIterations = 1000) {
    size_t n = A.size();
    // Ensure A is square.
    for (const auto &row : A)
        if (row.size() != n)
            throw std::invalid_argument("Matrix must be square for QR eigen decomposition.");

    // Initialize the eigenvector matrix as the identity.
    Matrix eigenvectors = identityMatrix(n);

    for (size_t iter = 0; iter < maxIterations; ++iter) {
        // Perform QR decomposition.
        auto [Q, R] = qrDecomposition(A);
        // Form the next iterate: A = R * Q.
        A = multiplyMatrices(R, Q);
        // Accumulate the orthogonal transformations.
        eigenvectors = multiplyMatrices(eigenvectors, Q);
        // Check convergence based on the off-diagonal Frobenius norm.
        if (offDiagonalFrobeniusNorm(A) < tolerance)
            break;
    }

    // The eigenvalues are approximated by the diagonal of A.
    Vector eigenvalues(n);
    for (size_t i = 0; i < n; ++i)
        eigenvalues[i] = A[i][i];

    return {eigenvalues, eigenvectors};
}

//------------------------------------------------------------------------------
// Utility function: printMatrix
// Prints a matrix to standard output.
void printMatrix(const Matrix &M) {
    for (const auto &row : M) {
        for (double val : row)
            std::cout << val << "\t";
        std::cout << "\n";
    }
}

//------------------------------------------------------------------------------
// Utility function: printVector
// Prints a vector to standard output.
void printVector(const Vector &v) {
    for (double val : v)
        std::cout << val << "\n";
}

//------------------------------------------------------------------------------
// Main function demonstrating the QR algorithm for general eigendecomposition.
int main() {
    try {
        // Define a sample non-symmetric 3x3 matrix.
        Matrix sampleMatrix = {
            { 4.0, -2.0,  1.0 },
            { 3.0,  6.0,  2.0 },
            { 2.0,  1.0,  3.0 }
        };

        std::cout << "Original Matrix:\n";
        printMatrix(sampleMatrix);
        std::cout << "\n";

        // Compute the eigendecomposition using the QR algorithm.
        auto [eigenvalues, eigenvectors] = qrEigenDecomposition(sampleMatrix, 1e-10, 1000);

        std::cout << "Eigenvalues (approximate):\n";
        printVector(eigenvalues);
        std::cout << "\nEigenvectors (approximate, each column is an eigenvector):\n";
        printMatrix(eigenvectors);
    }
    catch (const std::exception &ex) {
        std::cerr << "Error during eigendecomposition: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
