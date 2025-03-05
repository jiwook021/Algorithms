#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>

//------------------------------------------------------------------------------
// Function: computePNorm
// Description:
//   Computes the p-norm of a given container (e.g., vector) of numerical values.
//   The function works for any container that supports iteration over its elements.
//   The norm is defined as:
//       ||x||_p = (Σ |x_i|^p)^(1/p) for p != infinity, and
//       ||x||_∞ = max{|x_i|} for p = infinity.
// Parameters:
//   - vec: A constant reference to the container holding numerical values.
//   - p:   The order of the norm. Must be >= 1 or equal to infinity.
// Returns:
//   The computed p-norm as a double.
// Throws:
//   std::invalid_argument if p is not valid (i.e., p < 1 and not infinity).
// Time Complexity: O(n), where n is the number of elements in the container.
// Space Complexity: O(1)
// Thread Safety: This function does not modify any shared state, hence it is thread safe.
//------------------------------------------------------------------------------
template<typename Container>
double computePNorm(const Container& vec, double p) {
    // If the container is empty, return 0.0 as the norm.
    if (vec.empty()) {
        return 0.0;
    }
    
    // Check that p is valid: p must be >= 1 or p must be infinity.
    if (!std::isinf(p) && p < 1.0) {
        throw std::invalid_argument("The norm order p must be >= 1 or infinity.");
    }
    
    // If p is infinity, compute the maximum absolute value.
    if (std::isinf(p)) {
        double maxAbsoluteValue = 0.0;
        for (const auto& element : vec) {
            double currentAbsoluteValue = std::abs(element);
            if (currentAbsoluteValue > maxAbsoluteValue) {
                maxAbsoluteValue = currentAbsoluteValue;
            }
        }
        return maxAbsoluteValue;
    }
    
    // For finite p, compute the sum of |element|^p.
    double sumPowered = 0.0;
    for (const auto& element : vec) {
        sumPowered += std::pow(std::abs(element), p);
    }
    
    // Return the p-norm.
    return std::pow(sumPowered, 1.0 / p);
}

//------------------------------------------------------------------------------
// Main function demonstrating the usage of computePNorm.
//------------------------------------------------------------------------------
int main() {
    // Sample vector for which norms will be computed.
    std::vector<double> sampleVector {3.0, -4.0, 5.0, -2.0};

    try {
        // Compute L1 norm (p = 1)
        double normL1 = computePNorm(sampleVector, 1.0);
        std::cout << "L1 Norm: " << normL1 << std::endl;

        // Compute L2 norm (p = 2)
        double normL2 = computePNorm(sampleVector, 2.0);
        std::cout << "L2 Norm: " << normL2 << std::endl;

        // Compute L3 norm (p = 3)
        double normL3 = computePNorm(sampleVector, 3.0);
        std::cout << "L3 Norm: " << normL3 << std::endl;

        // Compute Infinity norm (p = infinity)
        double normInfinity = computePNorm(sampleVector, std::numeric_limits<double>::infinity());
        std::cout << "Infinity Norm: " << normInfinity << std::endl;

        // Edge case: computing norm for an empty vector.
        std::vector<double> emptyVector;
        double normEmpty = computePNorm(emptyVector, 2.0);
        std::cout << "Norm of empty vector: " << normEmpty << std::endl;

        // Edge case: p value less than 1 (should throw an exception).
        try {
            double invalidNorm = computePNorm(sampleVector, 0.5);
            std::cout << "This line should not execute: " << invalidNorm << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "Caught expected exception for invalid norm order: " << ex.what() << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "An error occurred while computing norms: " << ex.what() << std::endl;
    }

    return 0;
}
