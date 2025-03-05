#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>

// Using the Vector and Matrix classes from the original code
// Simple Vector class
class Vector {
private:
    std::vector<double> data;

public:
    // Default constructor
    Vector() : data() {}
    
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& vec) : data(vec) {}

    double& operator[](size_t index) { return data[index]; }
    const double& operator[](size_t index) const { return data[index]; }
    size_t size() const { return data.size(); }

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

    Vector operator-(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for subtraction");
        }
        
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    Vector operator*(double scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    double dot(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        
        double result = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            result += data[i] * other[i];
        }
        return result;
    }

    double mean() const {
        if (size() == 0) return 0.0;
        double sum = 0.0;
        for (const auto& val : data) {
            sum += val;
        }
        return sum / size();
    }

    double variance() const {
        if (size() <= 1) return 0.0;
        double m = mean();
        double sum_sq_diff = 0.0;
        for (const auto& val : data) {
            double diff = val - m;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / size();
    }

    double std_dev() const {
        return std::sqrt(variance());
    }

    // Pearson correlation coefficient with another vector
    double correlation(const Vector& other) const {
        if (size() != other.size() || size() == 0) {
            throw std::invalid_argument("Vectors must have the same non-zero size");
        }

        double mean_x = mean();
        double mean_y = other.mean();
        double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;

        for (size_t i = 0; i < size(); ++i) {
            double x_diff = data[i] - mean_x;
            double y_diff = other[i] - mean_y;
            sum_xy += x_diff * y_diff;
            sum_x2 += x_diff * x_diff;
            sum_y2 += y_diff * y_diff;
        }

        if (sum_x2 == 0.0 || sum_y2 == 0.0) {
            return 0.0;  // Avoid division by zero
        }

        return sum_xy / std::sqrt(sum_x2 * sum_y2);
    }

    const std::vector<double>& get_data() const { return data; }
};

// Simple Matrix class
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Default constructor
    Matrix() : data(), rows(0), cols(0) {}
    
    Matrix(size_t rows, size_t cols, double value = 0.0)
        : data(rows, std::vector<double>(cols, value)), rows(rows), cols(cols) {}

    Matrix(const std::vector<std::vector<double>>& mat) {
        rows = mat.size();
        cols = rows > 0 ? mat[0].size() : 0;
        data = mat;
    }

    std::vector<double>& operator[](size_t row) { return data[row]; }
    const std::vector<double>& operator[](size_t row) const { return data[row]; }

    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }

    Vector get_col(size_t col) const {
        if (col >= cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        Vector result(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i] = data[i][col];
        }
        return result;
    }

    // Get a specific row as a Vector
    Vector get_row(size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("Row index out of range");
        }
        
        return Vector(data[row]);
    }
};

// Feature scaling (min-max scaling)
std::pair<Matrix, std::vector<std::pair<double, double>>> scale_features(const Matrix& X) {
    size_t n_samples = X.num_rows();
    size_t n_features = X.num_cols();
    
    // Find min and max values for each feature
    std::vector<std::pair<double, double>> min_max(n_features);  // (min, max) pairs
    for (size_t j = 0; j < n_features; ++j) {
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        
        for (size_t i = 0; i < n_samples; ++i) {
            min_val = std::min(min_val, X[i][j]);
            max_val = std::max(max_val, X[i][j]);
        }
        
        min_max[j] = {min_val, max_val};
    }
    
    // Scale features
    Matrix X_scaled(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            double range = min_max[j].second - min_max[j].first;
            if (range > 0) {
                X_scaled[i][j] = (X[i][j] - min_max[j].first) / range;
            } else {
                X_scaled[i][j] = 0.5;  // Default value if min == max
            }
        }
    }
    
    return {X_scaled, min_max};
}

// Linear Regression with gradient descent
class LinearRegression {
private:
    Vector weights;
    double bias;
    double learning_rate;
    int max_iterations;
    double tol;
    bool verbose;
    
    // For feature scaling
    std::vector<std::pair<double, double>> feature_min_max;
    double target_min;
    double target_max;
    bool use_scaling;

public:
    LinearRegression(double learning_rate = 0.001, int max_iterations = 1000000, 
                     double tol = 1e-6, bool verbose = false, bool use_scaling = true)
        : learning_rate(learning_rate), max_iterations(max_iterations), 
          tol(tol), verbose(verbose), bias(0.0), use_scaling(use_scaling),
          target_min(0.0), target_max(1.0) {}

    void fit(const Matrix& X_orig, const Vector& y_orig) {
        if (X_orig.num_rows() != y_orig.size() || X_orig.num_rows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        // Feature scaling if enabled
        Matrix X;
        Vector y;
        
        if (use_scaling) {
            // Scale features
            auto [X_scaled, min_max] = scale_features(X_orig);
            X = X_scaled;
            feature_min_max = min_max;
            
            // Scale target
            target_min = *std::min_element(y_orig.get_data().begin(), y_orig.get_data().end());
            target_max = *std::max_element(y_orig.get_data().begin(), y_orig.get_data().end());
            
            double target_range = target_max - target_min;
            y = Vector(y_orig.size());
            for (size_t i = 0; i < y_orig.size(); ++i) {
                if (target_range > 0) {
                    y[i] = (y_orig[i] - target_min) / target_range;
                } else {
                    y[i] = 0.5;
                }
            }
        } else {
            X = X_orig;
            y = y_orig;
        }
        
        size_t n_samples = X.num_rows();
        size_t n_features = X.num_cols();

        // Initialize weights and bias
        weights = Vector(n_features, 0.0);
        bias = 0.0;

        double prev_loss = std::numeric_limits<double>::max();
        double curr_learning_rate = learning_rate;
        
        // Adaptive momentum
        Vector momentum(n_features, 0.0);
        double bias_momentum = 0.0;
        double beta = 0.9;  // Momentum factor

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Compute predictions
            Vector y_pred(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = bias;
                for (size_t j = 0; j < n_features; ++j) {
                    pred += X[i][j] * weights[j];
                }
                y_pred[i] = pred;
            }

            // Compute loss (MSE)
            double loss = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y[i];
                loss += error * error;
            }
            loss /= n_samples;

            // Check for convergence
            if (std::abs(loss - prev_loss) < tol) {
                if (verbose) {
                    std::cout << "Converged at iteration " << iter << " with loss " << loss << std::endl;
                }
                break;
            }
            
            // Learning rate scheduling
            // Reduce learning rate if loss increases
            if (loss > prev_loss) {
                curr_learning_rate *= 0.5;
                if (verbose) {
                    std::cout << "Reducing learning rate to " << curr_learning_rate << std::endl;
                }
            }
            
            prev_loss = loss;

            // Compute gradients
            Vector grad_w(n_features, 0.0);
            double grad_b = 0.0;

            for (size_t i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y[i];
                for (size_t j = 0; j < n_features; ++j) {
                    grad_w[j] += error * X[i][j];
                }
                grad_b += error;
            }

            // Scale gradients by number of samples
            for (size_t j = 0; j < n_features; ++j) {
                grad_w[j] /= n_samples;
            }
            grad_b /= n_samples;
            
            // Update with momentum
            for (size_t j = 0; j < n_features; ++j) {
                momentum[j] = beta * momentum[j] + (1.0 - beta) * grad_w[j];
                weights[j] -= curr_learning_rate * momentum[j];
            }
            bias_momentum = beta * bias_momentum + (1.0 - beta) * grad_b;
            bias -= curr_learning_rate * bias_momentum;
            
            // Debug output
            if (verbose && (iter % 1000 == 0 || iter == max_iterations - 1)) {
                std::cout << "Iteration " << iter << ": loss = " << loss << std::endl;
            }
        }
    }

    Vector predict(const Matrix& X_orig) const {
        if (X_orig.num_cols() != weights.size()) {
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        }
        
        // Scale input features if needed
        Matrix X;
        if (use_scaling && !feature_min_max.empty()) {
            size_t n_samples = X_orig.num_rows();
            size_t n_features = X_orig.num_cols();
            
            X = Matrix(n_samples, n_features);
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t j = 0; j < n_features; ++j) {
                    double range = feature_min_max[j].second - feature_min_max[j].first;
                    if (range > 0) {
                        X[i][j] = (X_orig[i][j] - feature_min_max[j].first) / range;
                    } else {
                        X[i][j] = 0.5;
                    }
                }
            }
        } else {
            X = X_orig;
        }
        
        size_t n_samples = X.num_rows();
        Vector y_pred(n_samples);

        // Make predictions
        for (size_t i = 0; i < n_samples; ++i) {
            double pred = bias;
            for (size_t j = 0; j < X.num_cols(); ++j) {
                pred += X[i][j] * weights[j];
            }
            y_pred[i] = pred;
        }
        
        // Unscale predictions if needed
        if (use_scaling) {
            double target_range = target_max - target_min;
            for (size_t i = 0; i < n_samples; ++i) {
                if (target_range > 0) {
                    y_pred[i] = y_pred[i] * target_range + target_min;
                } else {
                    y_pred[i] = target_min;
                }
            }
        }

        return y_pred;
    }

    Vector get_weights() const {
        return weights;
    }

    double get_bias() const {
        return bias;
    }

    // Calculate R-squared
    double r_squared(const Matrix& X, const Vector& y) const {
        if (X.num_rows() != y.size() || X.num_rows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        Vector y_pred = predict(X);
        double y_mean = y.mean();
        
        double ss_total = 0.0;
        double ss_residual = 0.0;
        
        for (size_t i = 0; i < y.size(); ++i) {
            double diff_total = y[i] - y_mean;
            double diff_residual = y[i] - y_pred[i];
            
            ss_total += diff_total * diff_total;
            ss_residual += diff_residual * diff_residual;
        }
        
        if (ss_total == 0.0) {
            return 0.0;  // Avoid division by zero
        }
        
        return 1.0 - (ss_residual / ss_total);
    }
};

// Multiple Linear Regression model (identical interface to LinearRegression)
typedef LinearRegression MultipleLinearRegression;

// Standardize features (z-score normalization)
Matrix standardize(const Matrix& X) {
    size_t n_samples = X.num_rows();
    size_t n_features = X.num_cols();
    
    Matrix X_std(n_samples, n_features);
    
    for (size_t j = 0; j < n_features; ++j) {
        Vector feature = X.get_col(j);
        double mean = feature.mean();
        double std_dev = feature.std_dev();
        
        for (size_t i = 0; i < n_samples; ++i) {
            if (std_dev > 0) {
                X_std[i][j] = (X[i][j] - mean) / std_dev;
            } else {
                X_std[i][j] = 0.0;
            }
        }
    }
    
    return X_std;
}

// Split data into training and testing sets
std::tuple<Matrix, Matrix, Vector, Vector> train_test_split(
    const Matrix& X, const Vector& y, double test_size = 0.2) {
    
    if (X.num_rows() != y.size() || X.num_rows() == 0) {
        throw std::invalid_argument("Invalid input data dimensions");
    }
    
    size_t n_samples = X.num_rows();
    size_t n_test = static_cast<size_t>(n_samples * test_size);
    
    // Ensure at least one test sample
    n_test = std::max(size_t(1), n_test);
    
    // Ensure at least one training sample
    size_t n_train = n_samples - n_test;
    n_train = std::max(size_t(1), n_train);
    
    // Adjust n_test if necessary
    n_test = n_samples - n_train;
    
    // Create indices and shuffle them
    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Split data
    Matrix X_train(n_train, X.num_cols());
    Matrix X_test(n_test, X.num_cols());
    Vector y_train(n_train);
    Vector y_test(n_test);
    
    for (size_t i = 0; i < n_train; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.num_cols(); ++j) {
            X_train[i][j] = X[idx][j];
        }
        y_train[i] = y[idx];
    }
    
    for (size_t i = 0; i < n_test; ++i) {
        size_t idx = indices[i + n_train];
        for (size_t j = 0; j < X.num_cols(); ++j) {
            X_test[i][j] = X[idx][j];
        }
        y_test[i] = y[idx];
    }
    
    return {X_train, X_test, y_train, y_test};
}

// Calculate Mean Squared Error
double mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / y_true.size();
}

int main() {
    std::cout << "===== Employee Performance Analysis with Machine Learning =====" << std::endl;
    
    // Define the employee performance data
    // Years of experience, Education level (1=Bachelor, 2=Master, 3=PhD), 
    // Number of completed projects, Average project delivery time (days),
    // Hours worked per week, Attendance rate (%), Score on aptitude test (0-100),
    // Annual performance evaluation score (0-100)
    std::vector<std::vector<double>> employee_data = {
        // YrsExp, EduLvl, Projects, DeliveryTime, HrsPerWeek, Attendance, AptitudeScore, PerfScore
        {2.0,     1.0,    4.0,      28.0,         38.0,       92.0,       78.0,          72.0},
        {5.0,     2.0,    8.0,      25.0,         42.0,       95.0,       82.0,          81.0},
        {1.0,     1.0,    2.0,      35.0,         35.0,       88.0,       70.0,          65.0},
        {8.0,     2.0,    12.0,     22.0,         45.0,       97.0,       85.0,          87.0},
        {3.0,     1.0,    6.0,      26.0,         40.0,       93.0,       75.0,          76.0},
        {10.0,    3.0,    15.0,     20.0,         48.0,       98.0,       90.0,          92.0},
        {4.0,     2.0,    7.0,      24.0,         41.0,       94.0,       80.0,          79.0},
        {0.5,     1.0,    1.0,      40.0,         35.0,       85.0,       65.0,          60.0},
        {6.0,     2.0,    10.0,     23.0,         44.0,       96.0,       84.0,          83.0},
        {12.0,    3.0,    18.0,     18.0,         50.0,       99.0,       95.0,          95.0},
        {7.0,     2.0,    11.0,     24.0,         43.0,       95.0,       83.0,          84.0},
        {2.5,     1.0,    5.0,      27.0,         39.0,       91.0,       76.0,          74.0},
        {9.0,     3.0,    14.0,     21.0,         47.0,       97.0,       88.0,          90.0},
        {1.5,     1.0,    3.0,      32.0,         37.0,       90.0,       72.0,          68.0},
        {4.5,     2.0,    8.0,      25.0,         42.0,       94.0,       81.0,          80.0},
        {11.0,    3.0,    17.0,     19.0,         49.0,       98.0,       92.0,          94.0},
        {3.5,     1.0,    6.0,      26.0,         40.0,       93.0,       77.0,          75.0},
        {8.5,     2.0,    13.0,     22.0,         46.0,       96.0,       86.0,          88.0},
        {6.5,     2.0,    10.0,     23.0,         44.0,       95.0,       84.0,          85.0},
        {2.0,     1.0,    3.0,      30.0,         38.0,       89.0,       74.0,          70.0}
    };
    
    // Extract features (X) and target (y) from the data
    size_t n_samples = employee_data.size();
    size_t n_features = 7;  // Number of features
    
    Matrix X(n_samples, n_features);
    Vector y(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X[i][j] = employee_data[i][j];  // All features
        }
        y[i] = employee_data[i][7];  // Performance score
    }
    
    // Feature names for reference
    std::vector<std::string> feature_names = {
        "Years of Experience", 
        "Education Level", 
        "Completed Projects", 
        "Avg Delivery Time (days)", 
        "Hours per Week", 
        "Attendance Rate (%)", 
        "Aptitude Test Score"
    };
    
    // Basic statistics
    std::cout << "\n===== Basic Statistics =====" << std::endl;
    std::cout << "Number of employees in dataset: " << n_samples << std::endl;
    
    for (size_t j = 0; j < n_features; ++j) {
        Vector feature = X.get_col(j);
        std::cout << "\n" << feature_names[j] << " Statistics:" << std::endl;
        std::cout << "  Mean: " << feature.mean() << std::endl;
        std::cout << "  Standard Deviation: " << feature.std_dev() << std::endl;
    }
    
    std::cout << "\nPerformance Score Statistics:" << std::endl;
    std::cout << "  Mean: " << y.mean() << std::endl;
    std::cout << "  Standard Deviation: " << y.std_dev() << std::endl;
    
    // Correlation analysis
    std::cout << "\n===== Correlation with Performance Score =====" << std::endl;
    std::cout << std::setw(30) << std::left << "Feature" 
              << std::setw(12) << std::right << "Correlation" << std::endl;
    std::cout << std::string(42, '-') << std::endl;
    
    std::vector<double> correlations;
    
    for (size_t j = 0; j < n_features; ++j) {
        Vector feature = X.get_col(j);
        double corr = feature.correlation(y);
        correlations.push_back(corr);
        
        std::cout << std::setw(30) << std::left << feature_names[j]
                  << std::setw(12) << std::right << std::fixed << std::setprecision(4) << corr << std::endl;
    }
    
    // Split data into training and testing sets
    // Using a fixed seed for reproducibility
    std::random_device rd;
    std::mt19937 g(42);  // Fixed seed
    
    // Create indices and shuffle them
    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Using 70% training, 30% testing
    size_t n_train = static_cast<size_t>(n_samples * 0.7);
    size_t n_test = n_samples - n_train;
    
    Matrix X_train(n_train, X.num_cols());
    Matrix X_test(n_test, X.num_cols());
    Vector y_train(n_train);
    Vector y_test(n_test);
    
    for (size_t i = 0; i < n_train; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.num_cols(); ++j) {
            X_train[i][j] = X[idx][j];
        }
        y_train[i] = y[idx];
    }
    
    for (size_t i = 0; i < n_test; ++i) {
        size_t idx = indices[i + n_train];
        for (size_t j = 0; j < X.num_cols(); ++j) {
            X_test[i][j] = X[idx][j];
        }
        y_test[i] = y[idx];
    }
    
    std::cout << "\n===== Linear Regression Models =====" << std::endl;
    std::cout << "Training set size: " << X_train.num_rows() << " samples" << std::endl;
    std::cout << "Test set size: " << X_test.num_rows() << " samples" << std::endl;
    
    // Train a multiple linear regression model with all features
    std::cout << "\nModel: Performance Score as a function of all features" << std::endl;
    
    try {
        MultipleLinearRegression model(0.05, 20000, 1e-6, false, true);
        model.fit(X_train, y_train);
        
        Vector y_pred = model.predict(X_test);
        double mse = mean_squared_error(y_test, y_pred);
        double r2 = model.r_squared(X_test, y_test);
        
        std::cout << "Model Performance:" << std::endl;
        std::cout << "  Mean Squared Error: " << mse << std::endl;
        std::cout << "  R-squared: " << r2 << std::endl;
        
        // Feature importance (based on standardized coefficients)
        Matrix X_std = standardize(X);
        MultipleLinearRegression std_model(0.05, 20000, 1e-6, false, true);
        std_model.fit(X_std, y);
        
        Vector std_coeffs = std_model.get_weights();
        
        std::cout << "\n===== Feature Importance Analysis =====" << std::endl;
        std::cout << std::setw(30) << std::left << "Feature" 
                  << std::setw(15) << std::right << "Coefficient" 
                  << std::setw(15) << std::right << "Std. Coefficient" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        Vector coeffs = model.get_weights();
        for (size_t j = 0; j < n_features; ++j) {
            std::cout << std::setw(30) << std::left << feature_names[j]
                      << std::setw(15) << std::right << std::fixed << std::setprecision(4) << coeffs[j]
                      << std::setw(15) << std::right << std::fixed << std::setprecision(4) << std_coeffs[j] << std::endl;
        }
        std::cout << std::setw(30) << std::left << "Intercept"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(4) << model.get_bias() << std::endl;
        
        // Performance prediction for different profiles
        std::cout << "\n===== Performance Predictions for Employee Profiles =====" << std::endl;
        
        struct EmployeeProfile {
            std::vector<double> features;
            std::string description;
        };
        
        std::vector<EmployeeProfile> profiles = {
            {{1.0, 1.0, 2.0, 30.0, 38.0, 90.0, 70.0}, "Junior, Bachelor's, Few Projects"},
            {{3.0, 1.0, 5.0, 25.0, 40.0, 93.0, 75.0}, "Mid-level, Bachelor's, Average Projects"},
            {{5.0, 2.0, 9.0, 22.0, 43.0, 95.0, 85.0}, "Experienced, Master's, Many Projects"},
            {{10.0, 3.0, 15.0, 20.0, 48.0, 98.0, 95.0}, "Senior, PhD, Numerous Projects"},
            {{2.0, 1.0, 3.0, 28.0, 45.0, 95.0, 80.0}, "Junior, Bachelor's, High Hours"},
            {{6.0, 2.0, 10.0, 21.0, 40.0, 97.0, 90.0}, "Experienced, Master's, High Aptitude"}
        };
        
        Matrix profile_matrix(profiles.size(), n_features);
        for (size_t i = 0; i < profiles.size(); ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                profile_matrix[i][j] = profiles[i].features[j];
            }
        }
        
        Vector predictions = model.predict(profile_matrix);
        
        std::cout << std::setw(40) << std::left << "Employee Profile" 
                  << std::setw(20) << std::right << "Predicted Score" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < profiles.size(); ++i) {
            std::cout << std::setw(40) << std::left << profiles[i].description
                      << std::setw(20) << std::right << std::fixed << std::setprecision(2) 
                      << predictions[i] << std::endl;
        }
        
        // Find the most influential positive and negative factors
        std::vector<std::pair<double, size_t>> feature_importance;
        for (size_t j = 0; j < n_features; ++j) {
            feature_importance.push_back({std::abs(std_coeffs[j]), j});
        }
        
        // Sort by absolute value (descending)
        std::sort(feature_importance.begin(), feature_importance.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "\n===== Most Influential Factors for Performance =====" << std::endl;
        std::cout << "Factors listed in order of importance:" << std::endl;
        
        for (size_t i = 0; i < feature_importance.size(); ++i) {
            size_t j = feature_importance[i].second;
            std::string effect = std_coeffs[j] > 0 ? "Positive" : "Negative";
            std::string explanation;
            
            // Custom explanations based on coefficient sign
            if (j == 0) { // Years of Experience
                explanation = std_coeffs[j] > 0 ? 
                    "More experienced employees tend to perform better" : 
                    "Years of experience may not translate to better performance";
            } else if (j == 1) { // Education Level
                explanation = std_coeffs[j] > 0 ? 
                    "Higher education correlates with better performance" : 
                    "Higher education doesn't necessarily lead to better performance";
            } else if (j == 2) { // Completed Projects
                explanation = std_coeffs[j] > 0 ? 
                    "Employees who complete more projects tend to score higher" : 
                    "Project quantity may be prioritized over quality";
            } else if (j == 3) { // Avg Delivery Time
                explanation = std_coeffs[j] > 0 ? 
                    "Longer delivery times correlate with better performance" : 
                    "Faster project completion correlates with better performance";
            } else if (j == 4) { // Hours per Week
                explanation = std_coeffs[j] > 0 ? 
                    "Employees who work more hours score higher" : 
                    "Working longer hours doesn't improve performance";
            } else if (j == 5) { // Attendance Rate
                explanation = std_coeffs[j] > 0 ? 
                    "Higher attendance correlates with better performance" : 
                    "Attendance doesn't significantly impact performance";
            } else if (j == 6) { // Aptitude Test Score
                explanation = std_coeffs[j] > 0 ? 
                    "Aptitude score strongly predicts job performance" : 
                    "Aptitude tests may not be relevant to job performance";
            }
            
            std::cout << i + 1 << ". " << feature_names[j] 
                      << " (Impact: " << effect << ")" << std::endl;
            std::cout << "   Coefficient: " << std_coeffs[j] << std::endl;
            std::cout << "   Interpretation: " << explanation << std::endl;
        }
        
        // Recommendations based on analysis
        std::cout << "\n===== Recommendations for Performance Improvement =====" << std::endl;
        
        // Look at top positive factors
        std::vector<size_t> top_positive;
        for (size_t j = 0; j < n_features; ++j) {
            if (std_coeffs[j] > 0) {
                top_positive.push_back(j);
            }
        }
        
        // Sort by coefficient value (descending)
        std::sort(top_positive.begin(), top_positive.end(),
            [&std_coeffs](size_t a, size_t b) { return std_coeffs[a] > std_coeffs[b]; });
        
        if (!top_positive.empty()) {
            std::cout << "Focus on improving these factors:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), top_positive.size()); ++i) {
                size_t j = top_positive[i];
                std::cout << "- " << feature_names[j] << std::endl;
            }
        }
        
        // Interactive prediction tool
        std::cout << "\n===== Interactive Performance Prediction Tool =====" << std::endl;
        std::cout << "Enter employee attributes to predict performance score." << std::endl;
        
        char continue_prediction = 'y';
        while (continue_prediction == 'y' || continue_prediction == 'Y') {
            Matrix input(1, n_features);
            
            std::cout << "\nEnter years of experience: ";
            std::cin >> input[0][0];
            
            std::cout << "Enter education level (1=Bachelor, 2=Master, 3=PhD): ";
            std::cin >> input[0][1];
            
            std::cout << "Enter number of completed projects: ";
            std::cin >> input[0][2];
            
            std::cout << "Enter average project delivery time (days): ";
            std::cin >> input[0][3];
            
            std::cout << "Enter hours worked per week: ";
            std::cin >> input[0][4];
            
            std::cout << "Enter attendance rate (%): ";
            std::cin >> input[0][5];
            
            std::cout << "Enter aptitude test score (0-100): ";
            std::cin >> input[0][6];
            
            Vector pred = model.predict(input);
            
            std::cout << "\nPredicted Performance Score: " << pred[0] << std::endl;
            
            // Show contribution of each feature
            std::cout << "\nContribution breakdown:" << std::endl;
            double base = model.get_bias();
            std::cout << "Base value: " << base << " points" << std::endl;
            
            for (size_t j = 0; j < n_features; ++j) {
                double contrib = coeffs[j] * input[0][j];
                std::cout << feature_names[j] << ": " << contrib << " points" << std::endl;
            }
            
            std::cout << "\nMake another prediction? (y/n): ";
            std::cin >> continue_prediction;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in model: " << e.what() << std::endl;
    }
    
    return 0;
}