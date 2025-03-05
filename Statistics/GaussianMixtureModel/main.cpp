#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Gaussian probability density function
double gaussian_pdf(double x, double mean, double variance) {
    const double PI = 3.141592653589793;
    double stddev = std::sqrt(variance);
    return (1.0 / (stddev * std::sqrt(2.0 * PI))) * std::exp(-std::pow(x - mean, 2) / (2.0 * variance));
}

// Structure to hold parameters for one Gaussian component
struct GaussianComponent {
    double mean;
    double variance;
    double weight;
};

// GMM class for two clusters
class GMM {
private:
    GaussianComponent comp1, comp2;          // Two Gaussian components
    std::vector<double> responsibilities1;   // P(cluster1 | x_i)
    std::vector<double> responsibilities2;   // P(cluster2 | x_i)
    double tolerance = 1e-4;                 // Convergence threshold
    int max_iterations = 100;                // Maximum EM iterations

public:
    // Initialize with random parameters
    GMM() {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        comp1 = {static_cast<double>(std::rand() % 100), 1.0, 0.5};
        comp2 = {static_cast<double>(std::rand() % 100), 1.0, 0.5};
    }

    // E-step: Compute responsibilities for each data point
    void e_step(const std::vector<double>& data) {
        responsibilities1.clear();
        responsibilities2.clear();
        for (double x : data) {
            double pdf1 = gaussian_pdf(x, comp1.mean, comp1.variance) * comp1.weight;
            double pdf2 = gaussian_pdf(x, comp2.mean, comp2.variance) * comp2.weight;
            double total = pdf1 + pdf2;
            responsibilities1.push_back(pdf1 / total);
            responsibilities2.push_back(pdf2 / total);
        }
    }

    // M-step: Update parameters based on responsibilities
    void m_step(const std::vector<double>& data) {
        double sum_resp1 = 0.0, sum_resp2 = 0.0;
        double weighted_sum1 = 0.0, weighted_sum2 = 0.0;
        double weighted_var1 = 0.0, weighted_var2 = 0.0;

        for (size_t i = 0; i < data.size(); ++i) {
            sum_resp1 += responsibilities1[i];
            sum_resp2 += responsibilities2[i];
            weighted_sum1 += responsibilities1[i] * data[i];
            weighted_sum2 += responsibilities2[i] * data[i];
            weighted_var1 += responsibilities1[i] * std::pow(data[i] - comp1.mean, 2);
            weighted_var2 += responsibilities2[i] * std::pow(data[i] - comp2.mean, 2);
        }

        // Update means
        comp1.mean = weighted_sum1 / sum_resp1;
        comp2.mean = weighted_sum2 / sum_resp2;

        // Update variances
        comp1.variance = weighted_var1 / sum_resp1;
        comp2.variance = weighted_var2 / sum_resp2;

        // Update weights
        comp1.weight = sum_resp1 / data.size();
        comp2.weight = sum_resp2 / data.size();
    }

    // Fit the GMM to the data using EM
    void fit(const std::vector<double>& data) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            double prev_mean1 = comp1.mean;
            double prev_mean2 = comp2.mean;

            e_step(data);
            m_step(data);

            // Check for convergence
            if (std::abs(comp1.mean - prev_mean1) < tolerance && std::abs(comp2.mean - prev_mean2) < tolerance) {
                std::cout << "Converged after " << iter + 1 << " iterations.\n";
                break;
            }
        }
    }

    // Predict the cluster for a data point
    int predict(double x) const {
        double prob1 = gaussian_pdf(x, comp1.mean, comp1.variance) * comp1.weight;
        double prob2 = gaussian_pdf(x, comp2.mean, comp2.variance) * comp2.weight;
        return (prob1 > prob2) ? 0 : 1;
    }

    // Display the learned parameters
    void print_parameters() const {
        std::cout << "Cluster 1: mean = " << comp1.mean << ", variance = " << comp1.variance << ", weight = " << comp1.weight << "\n";
        std::cout << "Cluster 2: mean = " << comp2.mean << ", variance = " << comp2.variance << ", weight = " << comp2.weight << "\n";
    }
};

int main() {
    // Hardcoded dataset with two distinct groups
    std::vector<double> data = {1.0, 2.0, 1.5, 2.5, 10.0, 11.0, 9.5, 10.5, 12.0};

    // Initialize and fit the GMM
    GMM model;
    model.fit(data);

    // Show the learned parameters
    model.print_parameters();

    // Display cluster assignments
    std::cout << "\nCluster assignments:\n";
    for (double x : data) {
        int cluster = model.predict(x);
        std::cout << "Value " << x << " -> Cluster " << cluster +1 << "\n";
    }

    return 0;
}