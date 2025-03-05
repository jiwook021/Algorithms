#include <iostream>
#include <vector>
#include <cmath>

// Structure for a 2D point
struct Point {
    double x;  // Feature 1
    double y;  // Feature 2
};

// PCA class
class PCA {
private:
    Point mean;           // Mean of the dataset
    double eigenvector_x; // x-component of the principal eigenvector
    double eigenvector_y; // y-component of the principal eigenvector

    // Compute the mean of the dataset
    Point compute_mean(const std::vector<Point>& data) const {
        Point m = {0.0, 0.0};
        for (const auto& p : data) {
            m.x += p.x;
            m.y += p.y;
        }
        m.x /= data.size();
        m.y /= data.size();
        return m;
    }

    // Compute the covariance matrix (2x2 for 2D data)
    std::vector<std::vector<double>> compute_covariance(const std::vector<Point>& data) const {
        std::vector<std::vector<double>> cov(2, std::vector<double>(2, 0.0));
        for (const auto& p : data) {
            double dx = p.x - mean.x;
            double dy = p.y - mean.y;
            cov[0][0] += dx * dx;
            cov[0][1] += dx * dy;
            cov[1][0] += dy * dx;
            cov[1][1] += dy * dy;
        }
        cov[0][0] /= data.size();
        cov[0][1] /= data.size();
        cov[1][0] /= data.size();
        cov[1][1] /= data.size();
        return cov;
    }

    // Simplified eigenvector computation for 2x2 matrix (first principal component)
    void compute_eigenvector(const std::vector<std::vector<double>>& cov) {
        // For a 2x2 matrix, solve the characteristic equation det(A - λI) = 0
        double a = cov[0][0], b = cov[0][1], c = cov[1][0], d = cov[1][1];
        double trace = a + d;
        double det = a * d - b * c;
        double discriminant = std::sqrt(trace * trace - 4 * det);
        double lambda1 = (trace + discriminant) / 2;  // Larger eigenvalue

        // Eigenvector for lambda1 (simplified for 2D)
        eigenvector_x = b;
        eigenvector_y = lambda1 - a;
        double norm = std::sqrt(eigenvector_x * eigenvector_x + eigenvector_y * eigenvector_y);
        eigenvector_x /= norm;
        eigenvector_y /= norm;
    }

public:
    // Train the PCA model
    void fit(const std::vector<Point>& data) {
        // Step 1: Compute the mean
        mean = compute_mean(data);

        // Step 2: Compute the covariance matrix
        auto cov = compute_covariance(data);

        // Step 3: Compute the principal eigenvector
        compute_eigenvector(cov);
    }

    // Project a point onto the first principal component
    double predict(const Point& p) const {
        // Center the point
        double dx = p.x - mean.x;
        double dy = p.y - mean.y;
        // Dot product with the principal eigenvector
        return dx * eigenvector_x + dy * eigenvector_y;
    }

    // Display the principal component direction
    void print_direction() const {
        std::cout << "Principal Component Direction: (" << eigenvector_x << ", " << eigenvector_y << ")\n";
    }
};

int main() {
    // Hardcoded dataset: 2D points with some correlation
    std::vector<Point> data = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
        {4.0, 5.0}, {5.0, 6.0}
    };

    // Initialize and train PCA
    PCA model;
    model.fit(data);

    // Show the direction of the principal component
    model.print_direction();

    // User input for a new point
    std::cout << "Enter x and y for a new point (e.g., 3.0 4.0): ";
    double x, y;
    std::cin >> x >> y;
    Point new_point = {x, y};

    // Project and display the result
    double projection = model.predict(new_point);
    std::cout << "Projected value for (" << x << ", " << y << "): " << projection << std::endl;

    return 0;
}