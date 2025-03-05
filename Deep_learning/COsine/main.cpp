#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Vector class definition
class Vector {
private:
    std::vector<double> data;

public:
    // Constructor with size, initialized to zeros
    Vector(size_t size) : data(size, 0.0) {}
    
    // Constructor from an existing std::vector
    Vector(const std::vector<double>& vec) : data(vec) {}
    
    // Get the size of the vector
    size_t size() const { return data.size(); }
    
    // Access and modify elements
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    
    // Vector addition
    Vector operator+(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for addition");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }
    
    // Scalar multiplication
    Vector operator*(double scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }
    
    // Dot product
    double dot(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            sum += data[i] * other[i];
        }
        return sum;
    }
    
    // Compute Euclidean norm (magnitude)
    double norm() const {
        return std::sqrt(dot(*this));
    }
    
    // Normalize the vector (returns a new vector)
    Vector normalize() const {
        double mag = norm();
        if (mag == 0.0) {
            throw std::invalid_argument("Cannot normalize a zero vector");
        }
        return (*this) * (1.0 / mag);
    }
};

// Function to compute cosine similarity between two vectors
double cosine_similarity(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size for cosine similarity");
    }
    double dot_product = v1.dot(v2);
    double norm_v1 = v1.norm();
    double norm_v2 = v2.norm();
    if (norm_v1 == 0.0 || norm_v2 == 0.0) {
        throw std::invalid_argument("Cannot compute cosine similarity with zero vector");
    }
    return dot_product / (norm_v1 * norm_v2);
}

// Main function to demonstrate vector normalization and cosine similarity
int main() {
    // Create two sample vectors
    Vector v1({1.0, 2.0, 3.0});
    Vector v2({4.0, 5.0, 6.0});
    
    // Compute and print cosine similarity
    try {
        double similarity = cosine_similarity(v1, v2);
        std::cout << "Cosine similarity between v1 and v2: " << similarity << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // Normalize v1 and print the result
    try {
        Vector v1_normalized = v1.normalize();
        std::cout << "Normalized v1: [";
        for (size_t i = 0; i < v1_normalized.size(); ++i) {
            std::cout << v1_normalized[i];
            if (i < v1_normalized.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Norm of normalized v1: " << v1_normalized.norm() << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}