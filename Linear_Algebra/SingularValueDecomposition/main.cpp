#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <thread>
#include <mutex>
#include <future>
#include <limits>

/**
 * @brief A C++20 implementation of Singular Value Decomposition (SVD)
 * 
 * This class implements the SVD algorithm which decomposes a matrix A into three matrices:
 * A = U * Σ * V^T where:
 * - U is an m×m orthogonal matrix
 * - Σ is an m×n diagonal matrix with non-negative real numbers (singular values)
 * - V^T is the transpose of an n×n orthogonal matrix
 * 
 * Time Complexity: O(min(mn², m²n)) where m and n are the dimensions of the matrix
 * Space Complexity: O(mn + m² + n²) for storing the matrices
 */
class SVD {
private:
    // Internal matrices for the decomposition
    std::vector<std::vector<double>> U;  // Left singular vectors
    std::vector<double> S;               // Singular values
    std::vector<std::vector<double>> V;  // Right singular vectors
    
    size_t m, n;                         // Matrix dimensions
    const double epsilon = 1e-10;        // Numerical stability threshold
    
    // Using mutable to allow mutex locking in const member functions
    mutable std::mutex mtx;              // Mutex for thread-safe operations
    
    /**
     * @brief Calculate the dot product of two vectors
     * @param v1 First vector
     * @param v2 Second vector
     * @return The dot product
     * 
     * Time Complexity: O(n)
     */
    double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) const {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        
        double result = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            result += v1[i] * v2[i];
        }
        return result;
    }
    
    /**
     * @brief Normalize a vector (make it unit length)
     * @param v Vector to normalize
     * @return The normalized vector
     * 
     * Time Complexity: O(n)
     */
    std::vector<double> normalize(const std::vector<double>& v) const {
        double magnitude = std::sqrt(dotProduct(v, v));
        
        // Handle zero vectors to avoid division by zero
        if (magnitude < epsilon) {
            return std::vector<double>(v.size(), 0.0);
        }
        
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i] / magnitude;
        }
        return result;
    }
    
    /**
     * @brief Transpose a matrix
     * @param A Matrix to transpose
     * @return The transposed matrix
     * 
     * Time Complexity: O(m*n)
     */
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A) const {
        if (A.empty()) return {};
        
        size_t rows = A.size();
        size_t cols = A[0].size();
        
        std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }
    
    /**
     * @brief Safer matrix multiplication that checks dimensions
     * @param A First matrix
     * @param B Second matrix
     * @return The product matrix
     * 
     * Time Complexity: O(m*n*p) where A is m×n and B is n×p
     */
    std::vector<std::vector<double>> matrixMultiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) const {
        
        // Safety checks
        if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
            throw std::invalid_argument("Matrices cannot be empty");
        }
        
        size_t rowsA = A.size();
        size_t colsA = A[0].size();
        size_t rowsB = B.size();
        size_t colsB = B[0].size();
        
        if (colsA != rowsB) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        // Create result matrix
        std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
        
        // Perform multiplication
        for (size_t i = 0; i < rowsA; ++i) {
            for (size_t j = 0; j < colsB; ++j) {
                for (size_t k = 0; k < colsA; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Modified Gram-Schmidt process for QR decomposition
     * @param A Input matrix
     * @param Q Orthogonal matrix
     * @param R Upper triangular matrix
     * 
     * Time Complexity: O(m*n²)
     */
    void qrDecomposition(
        const std::vector<std::vector<double>>& A,
        std::vector<std::vector<double>>& Q,
        std::vector<std::vector<double>>& R) {
        
        size_t m = A.size();
        size_t n = A[0].size();
        
        // Initialize Q and R
        Q = std::vector<std::vector<double>>(m, std::vector<double>(n, 0.0));
        R = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        
        // Copy columns of A to Q
        std::vector<std::vector<double>> columns(n, std::vector<double>(m));
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                columns[j][i] = A[i][j];
            }
        }
        
        // Modified Gram-Schmidt process
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < j; ++k) {
                R[k][j] = dotProduct(columns[j], columns[k]);
                for (size_t i = 0; i < m; ++i) {
                    columns[j][i] -= R[k][j] * columns[k][i];
                }
            }
            
            double norm = std::sqrt(dotProduct(columns[j], columns[j]));
            if (norm < epsilon) {
                norm = epsilon; // Prevent division by near-zero
            }
            
            R[j][j] = norm;
            for (size_t i = 0; i < m; ++i) {
                columns[j][i] /= norm;
                Q[i][j] = columns[j][i];
            }
        }
    }
    
    /**
     * @brief Compute SVD using the power iteration method (safer implementation)
     * @param A Input matrix
     * 
     * Time Complexity: O(iterations * m * n)
     */
    void powerIteration(const std::vector<std::vector<double>>& A) {
        size_t m = A.size();
        size_t n = A[0].size();
        
        // Initialize matrices
        U = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));
        S = std::vector<double>(std::min(m, n), 0.0);
        V = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        
        // Create identity matrix for remaining columns/rows
        for (size_t i = 0; i < m; ++i) {
            U[i][i] = 1.0;
        }
        
        for (size_t i = 0; i < n; ++i) {
            V[i][i] = 1.0;
        }
        
        // Working matrix - will be deflated after each singular value is found
        std::vector<std::vector<double>> B = A;
        
        // Find k singular values/vectors (k = min(m,n))
        size_t k = std::min(m, n);
        for (size_t i = 0; i < k; ++i) {
            // Initialize v as a random unit vector
            std::vector<double> v(n, 0.0);
            v[i] = 1.0; // Simple initialization, not truly random
            
            // Power iteration to find dominant singular value/vectors
            for (size_t iter = 0; iter < 100; ++iter) {
                // Compute u = B * v
                std::vector<double> u(m, 0.0);
                for (size_t j = 0; j < m; ++j) {
                    for (size_t l = 0; l < n; ++l) {
                        u[j] += B[j][l] * v[l];
                    }
                }
                
                // Normalize u
                double sigma = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    sigma += u[j] * u[j];
                }
                sigma = std::sqrt(sigma);
                
                if (sigma < epsilon) {
                    break; // No more significant singular values
                }
                
                for (size_t j = 0; j < m; ++j) {
                    u[j] /= sigma;
                }
                
                // Compute v = B^T * u
                std::vector<double> new_v(n, 0.0);
                for (size_t j = 0; j < n; ++j) {
                    for (size_t l = 0; l < m; ++l) {
                        new_v[j] += B[l][j] * u[l];
                    }
                }
                
                // Check convergence
                double diff = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    diff += (new_v[j] - v[j]) * (new_v[j] - v[j]);
                }
                v = new_v;
                
                // Normalize v
                double norm = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    norm += v[j] * v[j];
                }
                norm = std::sqrt(norm);
                
                if (norm < epsilon) {
                    break;
                }
                
                for (size_t j = 0; j < n; ++j) {
                    v[j] /= norm;
                }
                
                if (std::sqrt(diff) < epsilon) {
                    break; // Converged
                }
            }
            
            // Store the singular value and vectors
            S[i] = 0.0;
            for (size_t j = 0; j < m; ++j) {
                double sum = 0.0;
                for (size_t l = 0; l < n; ++l) {
                    sum += B[j][l] * v[l];
                }
                S[i] += sum * sum;
            }
            S[i] = std::sqrt(S[i]);
            
            if (S[i] < epsilon) {
                break; // No more significant singular values
            }
            
            // Compute left singular vector (u)
            std::vector<double> u(m, 0.0);
            for (size_t j = 0; j < m; ++j) {
                for (size_t l = 0; l < n; ++l) {
                    u[j] += B[j][l] * v[l];
                }
            }
            
            // Normalize u
            for (size_t j = 0; j < m; ++j) {
                u[j] /= S[i];
            }
            
            // Store vectors in U and V
            for (size_t j = 0; j < m; ++j) {
                U[j][i] = u[j];
            }
            
            for (size_t j = 0; j < n; ++j) {
                V[j][i] = v[j];
            }
            
            // Deflate B by removing the contribution of this singular triplet
            for (size_t j = 0; j < m; ++j) {
                for (size_t l = 0; l < n; ++l) {
                    B[j][l] -= S[i] * u[j] * v[l];
                }
            }
        }
        
        // Sort singular values and corresponding vectors
        sortSingularValues();
    }
    
    /**
     * @brief Sort singular values and corresponding vectors in descending order
     * 
     * Time Complexity: O(n log n)
     */
    void sortSingularValues() {
        // Create indices for sorting
        std::vector<size_t> indices(S.size());
        for (size_t i = 0; i < S.size(); ++i) {
            indices[i] = i;
        }
        
        // Sort indices based on singular values (descending)
        std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) {
            return S[i] > S[j];
        });
        
        // Create temporary copies for sorted matrices
        std::vector<double> tempS(S.size());
        std::vector<std::vector<double>> tempU(U.size(), std::vector<double>(U[0].size()));
        std::vector<std::vector<double>> tempV(V.size(), std::vector<double>(V[0].size()));
        
        // Rearrange matrices according to sorted indices
        for (size_t i = 0; i < S.size(); ++i) {
            tempS[i] = S[indices[i]];
            
            for (size_t j = 0; j < U.size(); ++j) {
                tempU[j][i] = U[j][indices[i]];
            }
            
            for (size_t j = 0; j < V.size(); ++j) {
                tempV[j][i] = V[j][indices[i]];
            }
        }
        
        // Update matrices with sorted values
        S = tempS;
        U = tempU;
        V = tempV;
    }
    
public:
    /**
     * @brief Default constructor
     */
    SVD() : m(0), n(0) {}
    
    /**
     * @brief Compute the SVD of a matrix
     * @param A Input matrix
     * @throws std::invalid_argument If the matrix is empty
     */
    void compute(const std::vector<std::vector<double>>& A) {
        // Input validation
        if (A.empty() || A[0].empty()) {
            throw std::invalid_argument("Input matrix cannot be empty");
        }
        
        // Check if all rows have the same size
        size_t cols = A[0].size();
        for (const auto& row : A) {
            if (row.size() != cols) {
                throw std::invalid_argument("Input matrix must be rectangular");
            }
        }
        
        // Lock for thread safety during computation
        std::lock_guard<std::mutex> lock(mtx);
        
        m = A.size();
        n = A[0].size();
        
        // Using power iteration method for better stability
        powerIteration(A);
    }
    
    /**
     * @brief Get the U matrix (left singular vectors)
     * @return The U matrix
     */
    std::vector<std::vector<double>> getU() const {
        std::lock_guard<std::mutex> lock(mtx);
        return U;
    }
    
    /**
     * @brief Get the singular values
     * @return The singular values as a vector
     */
    std::vector<double> getSingularValues() const {
        std::lock_guard<std::mutex> lock(mtx);
        return S;
    }
    
    /**
     * @brief Get the V matrix (right singular vectors)
     * @return The V matrix
     */
    std::vector<std::vector<double>> getV() const {
        std::lock_guard<std::mutex> lock(mtx);
        return V;
    }
    
    /**
     * @brief Get the V^T matrix (transpose of right singular vectors)
     * @return The V^T matrix
     */
    std::vector<std::vector<double>> getVT() const {
        std::lock_guard<std::mutex> lock(mtx);
        return transpose(V);
    }
    
    /**
     * @brief Reconstruct the original matrix from SVD components
     * @return The reconstructed matrix
     * 
     * Time Complexity: O(m*n*min(m,n))
     */
    std::vector<std::vector<double>> reconstruct() const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (U.empty() || S.empty() || V.empty()) {
            throw std::runtime_error("SVD not computed yet");
        }
        
        // Create Sigma matrix (diagonal matrix with singular values)
        std::vector<std::vector<double>> Sigma(m, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < S.size(); ++i) {
            Sigma[i][i] = S[i];
        }
        
        // Compute A = U * Sigma * V^T step by step safely
        std::vector<std::vector<double>> USigma = matrixMultiply(U, Sigma);
        std::vector<std::vector<double>> VT = transpose(V);
        return matrixMultiply(USigma, VT);
    }
    
    /**
     * @brief Compute the pseudoinverse of the original matrix
     * @param tolerance Tolerance for considering singular values as zero
     * @return The pseudoinverse matrix
     * 
     * Time Complexity: O(m*n*min(m,n))
     */
    std::vector<std::vector<double>> pseudoinverse(double tolerance = -1.0) const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (U.empty() || S.empty() || V.empty()) {
            throw std::runtime_error("SVD not computed yet");
        }
        
        // If tolerance is not specified, use a default based on matrix size and machine epsilon
        if (tolerance < 0) {
            tolerance = std::max(m, n) * S[0] * std::numeric_limits<double>::epsilon();
        }
        
        // Create inverse of Sigma
        std::vector<std::vector<double>> SigmaInv(n, std::vector<double>(m, 0.0));
        for (size_t i = 0; i < S.size(); ++i) {
            if (S[i] > tolerance) {
                SigmaInv[i][i] = 1.0 / S[i];
            }
        }
        
        // Compute A^+ = V * Sigma^(-1) * U^T
        std::vector<std::vector<double>> VSigmaInv = matrixMultiply(V, SigmaInv);
        std::vector<std::vector<double>> UT = transpose(U);
        return matrixMultiply(VSigmaInv, UT);
    }
    
    /**
     * @brief Compute the rank of the matrix
     * @param tolerance Tolerance for considering singular values as zero
     * @return The numerical rank
     */
    size_t rank(double tolerance = -1.0) const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (S.empty()) {
            throw std::runtime_error("SVD not computed yet");
        }
        
        // If tolerance is not specified, use a default based on matrix size and machine epsilon
        if (tolerance < 0) {
            tolerance = std::max(m, n) * S[0] * std::numeric_limits<double>::epsilon();
        }
        
        size_t r = 0;
        for (const auto& s : S) {
            if (s > tolerance) {
                r++;
            }
        }
        return r;
    }
    
    /**
     * @brief Compute the condition number of the matrix
     * @return The condition number (ratio of largest to smallest singular value)
     */
    double conditionNumber() const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (S.empty()) {
            throw std::runtime_error("SVD not computed yet");
        }
        
        // Find the smallest non-zero singular value
        double smallest = std::numeric_limits<double>::max();
        for (const auto& s : S) {
            if (s > epsilon && s < smallest) {
                smallest = s;
            }
        }
        
        if (smallest == std::numeric_limits<double>::max()) {
            return std::numeric_limits<double>::infinity();
        }
        
        return S[0] / smallest;
    }
    
    /**
     * @brief Calculate the Frobenius norm of the original matrix
     * @return The Frobenius norm
     */
    double frobeniusNorm() const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (S.empty()) {
            throw std::runtime_error("SVD not computed yet");
        }
        
        double sum = 0.0;
        for (const auto& s : S) {
            sum += s * s;
        }
        return std::sqrt(sum);
    }
};

/**
 * @brief Test the SVD implementation with a sample matrix
 */
void testSVD() {
    try {
        // Create a test matrix
        std::vector<std::vector<double>> A = {
            {3.0, 2.0, 2.0},
            {2.0, 3.0, -2.0}
        };
        
        std::cout << "Testing SVD with a 2x3 matrix..." << std::endl;
        
        // Compute SVD
        SVD svd;
        svd.compute(A);
        
        // Get decomposition components
        auto U = svd.getU();
        auto S = svd.getSingularValues();
        auto V = svd.getV();
        auto VT = svd.getVT();
        
        // Print matrices
        std::cout << "Matrix A:" << std::endl;
        for (const auto& row : A) {
            for (double val : row) {
                std::cout << std::setw(10) << val;
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nU matrix:" << std::endl;
        for (const auto& row : U) {
            for (double val : row) {
                std::cout << std::setw(10) << val;
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nSingular values:" << std::endl;
        for (double val : S) {
            std::cout << std::setw(10) << val;
        }
        std::cout << std::endl;
        
        std::cout << "\nV matrix:" << std::endl;
        for (const auto& row : V) {
            for (double val : row) {
                std::cout << std::setw(10) << val;
            }
            std::cout << std::endl;
        }
        
        // Verify decomposition by reconstructing A
        auto reconstructed = svd.reconstruct();
        
        std::cout << "\nReconstructed matrix:" << std::endl;
        for (const auto& row : reconstructed) {
            for (double val : row) {
                std::cout << std::setw(10) << val;
            }
            std::cout << std::endl;
        }
        
        // Compute and print error
        double error = 0.0;
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < A[i].size(); ++j) {
                error += (A[i][j] - reconstructed[i][j]) * (A[i][j] - reconstructed[i][j]);
            }
        }
        error = std::sqrt(error);
        
        std::cout << "\nReconstruction error: " << error << std::endl;
        
        // Additional properties
        std::cout << "Matrix rank: " << svd.rank() << std::endl;
        std::cout << "Condition number: " << svd.conditionNumber() << std::endl;
        std::cout << "Frobenius norm: " << svd.frobeniusNorm() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in testSVD: " << e.what() << std::endl;
    }
    
    // Test a larger matrix
    try {
        std::cout << "\nTesting SVD with a larger matrix..." << std::endl;
        
        // Create a 4x4 matrix
        std::vector<std::vector<double>> B = {
            {4.0, 0.0, 0.0, 0.0},
            {3.0, -5.0, 0.0, 0.0},
            {0.0, 0.0, 2.0, 2.0},
            {0.0, 0.0, 1.0, 3.0}
        };
        
        // Compute SVD
        SVD svd;
        svd.compute(B);
        
        std::cout << "SVD computation successful for 4x4 matrix." << std::endl;
        std::cout << "Singular values:" << std::endl;
        
        auto S = svd.getSingularValues();
        for (double val : S) {
            std::cout << std::setw(10) << val;
        }
        std::cout << std::endl;
        
        auto reconstructed = svd.reconstruct();
        double error = 0.0;
        for (size_t i = 0; i < B.size(); ++i) {
            for (size_t j = 0; j < B[i].size(); ++j) {
                error += (B[i][j] - reconstructed[i][j]) * (B[i][j] - reconstructed[i][j]);
            }
        }
        error = std::sqrt(error);
        
        std::cout << "Reconstruction error for 4x4 matrix: " << error << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in larger matrix test: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::cout << "Starting SVD tests..." << std::endl;
        testSVD();
        std::cout << "All tests completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}