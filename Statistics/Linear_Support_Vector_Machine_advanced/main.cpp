#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <string>

/**
 * DataPoint structure - represents a data point with two features
 * and a binary label for classification
 */
struct DataPoint {
    double x1;    // Feature 1
    double x2;    // Feature 2
    int label;    // Binary label (-1 or 1)

    // Constructor
    DataPoint(double x1, double x2, int label) : x1(x1), x2(x2), label(label) {}
};

/**
 * Data Preprocessor class - performs feature scaling
 * (using Z-score normalization to make mean=0, std=1)
 */
class DataPreprocessor {
private:
    bool is_fitted = false;
    double x1_mean = 0.0, x2_mean = 0.0;
    double x1_std = 1.0, x2_std = 1.0;

public:
    // Fit the preprocessor to the training data (calculate mean, std)
    void fit(const std::vector<DataPoint>& dataset) {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset is empty");
        }

        // Calculate means
        for (const auto& dp : dataset) {
            x1_mean += dp.x1;
            x2_mean += dp.x2;
        }
        x1_mean /= dataset.size();
        x2_mean /= dataset.size();

        // Calculate standard deviations
        for (const auto& dp : dataset) {
            x1_std += (dp.x1 - x1_mean) * (dp.x1 - x1_mean);
            x2_std += (dp.x2 - x2_mean) * (dp.x2 - x2_mean);
        }
        x1_std = std::sqrt(x1_std / dataset.size());
        x2_std = std::sqrt(x2_std / dataset.size());

        // Prevent division by zero
        x1_std = (x1_std < 1e-10) ? 1.0 : x1_std;
        x2_std = (x2_std < 1e-10) ? 1.0 : x2_std;

        is_fitted = true;
    }

    // Transform data to normalized form
    std::vector<DataPoint> transform(const std::vector<DataPoint>& dataset) const {
        if (!is_fitted) {
            throw std::runtime_error("Preprocessor must be fitted before transforming");
        }

        std::vector<DataPoint> normalized_data;
        normalized_data.reserve(dataset.size());

        for (const auto& dp : dataset) {
            double norm_x1 = (dp.x1 - x1_mean) / x1_std;
            double norm_x2 = (dp.x2 - x2_mean) / x2_std;
            normalized_data.emplace_back(norm_x1, norm_x2, dp.label);
        }

        return normalized_data;
    }

    // Normalize a single data point
    std::pair<double, double> transform_point(double x1, double x2) const {
        if (!is_fitted) {
            throw std::runtime_error("Preprocessor must be fitted before transforming");
        }
        return {(x1 - x1_mean) / x1_std, (x2 - x2_mean) / x2_std};
    }

    // Print preprocessor parameters
    void print_parameters() const {
        if (!is_fitted) {
            throw std::runtime_error("Preprocessor has not been fitted");
        }
        std::cout << "Preprocessing parameters:" << std::endl;
        std::cout << "  x1_mean = " << x1_mean << ", x1_std = " << x1_std << std::endl;
        std::cout << "  x2_mean = " << x2_mean << ", x2_std = " << x2_std << std::endl;
    }
};

/**
 * Kernel interface - supports non-linear SVM
 */
class Kernel {
public:
    virtual ~Kernel() = default;
    virtual double compute(const DataPoint& dp1, const DataPoint& dp2) const = 0;
    virtual std::string name() const = 0;
};

/**
 * Linear Kernel: K(x,y) = x·y
 */
class LinearKernel : public Kernel {
public:
    double compute(const DataPoint& dp1, const DataPoint& dp2) const override {
        return dp1.x1 * dp2.x1 + dp1.x2 * dp2.x2;
    }

    std::string name() const override {
        return "Linear";
    }
};

/**
 * RBF (Gaussian) Kernel: K(x,y) = exp(-gamma * ||x-y||^2)
 * Supports non-linear decision boundaries
 */
class RBFKernel : public Kernel {
private:
    double gamma;  // RBF kernel parameter

public:
    explicit RBFKernel(double gamma) : gamma(gamma) {
        if (gamma <= 0.0) {
            throw std::invalid_argument("gamma must be positive");
        }
    }

    double compute(const DataPoint& dp1, const DataPoint& dp2) const override {
        double squared_distance = 
            (dp1.x1 - dp2.x1) * (dp1.x1 - dp2.x1) + 
            (dp1.x2 - dp2.x2) * (dp1.x2 - dp2.x2);
        return std::exp(-gamma * squared_distance);
    }

    std::string name() const override {
        return "RBF (gamma=" + std::to_string(gamma) + ")";
    }
};

/**
 * Improved SVM class (using SMO algorithm)
 * Time Complexity: O(n²) for kernel matrix computation, O(n²×i) for SMO algorithm (n=data size, i=iterations)
 * Space Complexity: O(n²) for kernel matrix storage, O(s) for support vector storage (s=num support vectors)
 */
class ImprovedSVM {
private:
    // Model parameters
    std::vector<double> alphas;  // Lagrange multipliers
    std::vector<DataPoint> support_vectors;  // Support vectors
    double b = 0.0;  // Bias term

    // Training parameters
    double reg_strength;  // Regularization strength (C parameter, higher means weaker regularization)
    double tolerance;     // Convergence tolerance
    int max_passes;       // Maximum passes without alpha changes
    int max_iterations;   // Maximum total iterations
    double epsilon;       // Numerical tolerance

    // Kernel
    std::unique_ptr<Kernel> kernel;

    // Training metrics
    int iterations_performed = 0;
    double training_accuracy = 0.0;

    // Random number generator
    std::mt19937 rng{std::random_device{}()};

    // Helper function to calculate decision function
    double calculate_decision_function(const DataPoint& point) const {
        double result = 0.0;
        for (size_t i = 0; i < support_vectors.size(); ++i) {
            result += alphas[i] * support_vectors[i].label * 
                     kernel->compute(support_vectors[i], point);
        }
        return result + b;
    }

public:
    // Constructor with configurable parameters
    ImprovedSVM(double reg_strength = 1.0, double tolerance = 0.001, int max_passes = 5, 
        int max_iterations = 1000, double epsilon = 1e-8, 
        std::unique_ptr<Kernel> kernel_ptr = std::make_unique<LinearKernel>())
        : reg_strength(reg_strength), tolerance(tolerance), max_passes(max_passes), 
          max_iterations(max_iterations), epsilon(epsilon), 
          kernel(std::move(kernel_ptr)) {
        
        // Parameter validation
        if (reg_strength <= 0) throw std::invalid_argument("Regularization strength must be positive");
        if (tolerance <= 0) throw std::invalid_argument("Tolerance must be positive");
        if (max_passes <= 0) throw std::invalid_argument("Max passes must be positive");
        if (max_iterations <= 0) throw std::invalid_argument("Max iterations must be positive");
        if (epsilon <= 0) throw std::invalid_argument("Epsilon must be positive");
        if (!kernel) throw std::invalid_argument("Kernel cannot be null");
    }

    // Train SVM using SMO algorithm
    void train(const std::vector<DataPoint>& dataset) {
        if (dataset.empty()) {
            throw std::invalid_argument("Training dataset is empty");
        }

        const int n = static_cast<int>(dataset.size());
        alphas.resize(n, 0.0);
        std::vector<double> errors(n, 0.0);

        // Initialize algorithm variables
        int passes = 0;
        iterations_performed = 0;

        // Pre-compute kernel matrix for efficiency
        std::vector<std::vector<double>> kernel_matrix(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                kernel_matrix[i][j] = kernel->compute(dataset[i], dataset[j]);
                kernel_matrix[j][i] = kernel_matrix[i][j];  // Symmetry
            }
        }

        // Main SMO loop
        while ((passes < max_passes) && (iterations_performed < max_iterations)) {
            int num_changed_alphas = 0;

            // For each training example
            for (int i = 0; i < n; ++i) {
                // Calculate error for this example
                double Ei = 0.0;
                for (int j = 0; j < n; ++j) {
                    Ei += alphas[j] * dataset[j].label * kernel_matrix[i][j];
                }
                Ei += b - dataset[i].label;
                errors[i] = Ei;

                // Check if example violates KKT conditions
                bool kkt_violated = (dataset[i].label * Ei < -tolerance && alphas[i] < reg_strength) || 
                                  (dataset[i].label * Ei > tolerance && alphas[i] > 0);

                if (kkt_violated) {
                    // Select second example to optimize
                    int j = i;
                    while (j == i) {
                        j = std::uniform_int_distribution<int>(0, n-1)(rng);
                    }

                    // Calculate error for second example
                    double Ej = 0.0;
                    for (int k = 0; k < n; ++k) {
                        Ej += alphas[k] * dataset[k].label * kernel_matrix[j][k];
                    }
                    Ej += b - dataset[j].label;
                    errors[j] = Ej;

                    // Store old alphas
                    double alpha_i_old = alphas[i];
                    double alpha_j_old = alphas[j];

                    // Compute bounds for alpha_j
                    double L, H;
                    if (dataset[i].label != dataset[j].label) {
                        L = std::max(0.0, alphas[j] - alphas[i]);
                        H = std::min(reg_strength, reg_strength + alphas[j] - alphas[i]);
                    } else {
                        L = std::max(0.0, alphas[i] + alphas[j] - reg_strength);
                        H = std::min(reg_strength, alphas[i] + alphas[j]);
                    }

                    if (std::abs(L - H) < epsilon) {
                        continue;  // Skip if bounds are too close
                    }

                    // Compute eta
                    double eta = 2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j];
                    
                    if (eta >= 0) {
                        continue;  // Skip if eta is not negative
                    }

                    // Update alpha_j
                    alphas[j] = alpha_j_old - dataset[j].label * (Ei - Ej) / eta;

                    // Clip alpha_j to bounds
                    alphas[j] = std::min(H, std::max(L, alphas[j]));

                    if (std::abs(alphas[j] - alpha_j_old) < epsilon) {
                        continue;  // Skip if alpha_j didn't change much
                    }

                    // Update alpha_i
                    alphas[i] = alpha_i_old + dataset[i].label * dataset[j].label * 
                               (alpha_j_old - alphas[j]);

                    // Update threshold b
                    double b1 = b - Ei - dataset[i].label * (alphas[i] - alpha_i_old) * kernel_matrix[i][i] -
                               dataset[j].label * (alphas[j] - alpha_j_old) * kernel_matrix[i][j];
                    
                    double b2 = b - Ej - dataset[i].label * (alphas[i] - alpha_i_old) * kernel_matrix[i][j] -
                               dataset[j].label * (alphas[j] - alpha_j_old) * kernel_matrix[j][j];

                    if (0 < alphas[i] && alphas[i] < reg_strength) {
                        b = b1;
                    } else if (0 < alphas[j] && alphas[j] < reg_strength) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2.0;
                    }

                    num_changed_alphas++;
                }
            }

            iterations_performed++;

            if (num_changed_alphas == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }

        // Extract support vectors
        support_vectors.clear();
        std::vector<double> sv_alphas;
        
        for (int i = 0; i < n; ++i) {
            if (alphas[i] > epsilon) {  // Non-zero alpha indicates a support vector
                support_vectors.push_back(dataset[i]);
                sv_alphas.push_back(alphas[i]);
            }
        }
        
        // Update alphas to match support vectors
        alphas = sv_alphas;

        // Calculate training accuracy
        int correct = 0;
        for (const auto& dp : dataset) {
            if (predict(dp.x1, dp.x2) == dp.label) {
                correct++;
            }
        }
        training_accuracy = static_cast<double>(correct) / n;
    }

    // Predict class for a data point: 1 if decision function >= 0, else -1
    int predict(double x1, double x2) const {
        if (support_vectors.empty()) {
            throw std::runtime_error("Model has not been trained yet");
        }
        
        DataPoint point(x1, x2, 0);  // Label doesn't matter for prediction
        return calculate_decision_function(point) >= 0 ? 1 : -1;
    }

    // Calculate decision function value (distance from hyperplane)
    double decision_function(double x1, double x2) const {
        if (support_vectors.empty()) {
            throw std::runtime_error("Model has not been trained yet");
        }
        
        DataPoint point(x1, x2, 0);
        return calculate_decision_function(point);
    }

    // Print training statistics
    void print_statistics() const {
        std::cout << "Training completed in " << iterations_performed << " iterations." << std::endl;
        std::cout << "Number of support vectors: " << support_vectors.size() << std::endl;
        std::cout << "Training accuracy: " << std::fixed << std::setprecision(2) 
                 << (training_accuracy * 100) << "%" << std::endl;
        std::cout << "Bias term (b): " << b << std::endl;
        std::cout << "Kernel type: " << kernel->name() << std::endl;
    }
};

int main() {
    try {
        // Create dataset (same as original)
        std::vector<DataPoint> dataset = {
            {2.0, 3.0, -1},  // Class -1
            {3.0, 3.0, -1},
            {3.0, 4.0, -1},
            {5.0, 5.0, 1},   // Class 1
            {6.0, 5.0, 1},
            {7.0, 6.0, 1}
        };

        // Preprocess data
        DataPreprocessor preprocessor;
        preprocessor.fit(dataset);
        auto normalized_dataset = preprocessor.transform(dataset);
        
        // Print preprocessing parameters
        preprocessor.print_parameters();

        // Select kernel type
        std::cout << "Select kernel type:" << std::endl;
        std::cout << "1. Linear" << std::endl;
        std::cout << "2. RBF (Gaussian)" << std::endl;
        std::cout << "Enter choice (1 or 2): ";
        
        int kernel_choice;
        std::cin >> kernel_choice;
        
        std::unique_ptr<Kernel> kernel;
        if (kernel_choice == 1) {
            kernel = std::make_unique<LinearKernel>();
        } else if (kernel_choice == 2) {
            std::cout << "Enter gamma parameter for RBF kernel (suggested: 0.1-10): ";
            double gamma;
            std::cin >> gamma;
            kernel = std::make_unique<RBFKernel>(gamma);
        } else {
            throw std::invalid_argument("Invalid kernel selection");
        }

        // Initialize SVM with improved parameters
        std::cout << "Enter regularization parameter C (suggested: 0.1-10, lower means stronger regularization): ";
        double C;
        std::cin >> C;
        
        ImprovedSVM model(
            C,                      // Regularization parameter
            0.001,                  // Convergence tolerance
            5,                      // Maximum passes without changes
            1000,                   // Maximum iterations
            1e-8,                   // Numerical tolerance
            std::move(kernel)       // Kernel function
        );

        // Train the model
        std::cout << "Training model..." << std::endl;
        model.train(normalized_dataset);

        // Print training statistics
        model.print_statistics();

        // User input for prediction
        char continue_prediction;
        do {
            std::cout << "\nEnter x1 and x2 values for prediction (e.g., 4.0 4.0): ";
            double x1, x2;
            std::cin >> x1 >> x2;

            try {
                // Preprocess input
                auto [normalized_x1, normalized_x2] = preprocessor.transform_point(x1, x2);
                
                // Make prediction
                int prediction = model.predict(normalized_x1, normalized_x2);
                double decision_value = model.decision_function(normalized_x1, normalized_x2);
                
                std::cout << "Input point: (" << x1 << ", " << x2 << ")" << std::endl;
                std::cout << "Normalized point: (" << normalized_x1 << ", " << normalized_x2 << ")" << std::endl;
                std::cout << "Decision function value: " << decision_value << std::endl;
                std::cout << "Predicted class: " << prediction << " (confidence: " 
                         << std::abs(decision_value) << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Prediction error: " << e.what() << std::endl;
            }
            
            std::cout << "Predict another point? (y/n): ";
            std::cin >> continue_prediction;
        } while (continue_prediction == 'y' || continue_prediction == 'Y');

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}