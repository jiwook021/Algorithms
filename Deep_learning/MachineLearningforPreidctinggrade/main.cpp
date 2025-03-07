#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>

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

// Unscale a value using the min-max values
double unscale_value(double scaled_value, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range > 0) {
        return scaled_value * range + min_val;
    } else {
        return min_val;
    }
}

// Scale a value using the min-max values
double scale_value(double value, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range > 0) {
        return (value - min_val) / range;
    } else {
        return 0.5;  // Default value if min == max
    }
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
                std::cout << "  weights = [";
                for (size_t j = 0; j < n_features; ++j) {
                    std::cout << weights[j];
                    if (j < n_features - 1) std::cout << ", ";
                }
                std::cout << "], bias = " << bias << std::endl;
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
        if (!use_scaling || feature_min_max.empty()) {
            return weights;
        }
        
        // Transform weights to original scale
        Vector orig_weights(weights.size());
        double target_range = target_max - target_min;
        
        for (size_t j = 0; j < weights.size(); ++j) {
            double feature_range = feature_min_max[j].second - feature_min_max[j].first;
            if (feature_range > 0 && target_range > 0) {
                orig_weights[j] = weights[j] * target_range / feature_range;
            } else {
                orig_weights[j] = weights[j];
            }
        }
        
        return orig_weights;
    }

    double get_bias() const {
        if (!use_scaling) {
            return bias;
        }
        
        // Transform bias to original scale
        double target_range = target_max - target_min;
        double unscaled_bias = bias * target_range + target_min;
        
        // Adjust for feature scaling in the weights
        for (size_t j = 0; j < weights.size(); ++j) {
            double feature_min = feature_min_max[j].first;
            double feature_range = feature_min_max[j].second - feature_min;
            
            if (feature_range > 0) {
                unscaled_bias -= weights[j] * target_range * feature_min / feature_range;
            }
        }
        
        return unscaled_bias;
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
    std::cout << "===== IQ and Study Time Analysis with Machine Learning =====" << std::endl;
    int lowest_IQ = 70; 
    // Define the data directly in the code
    std::vector<std::vector<double>> student_data = {
        // IQ,   StudyTime, Grade
        {105.0-lowest_IQ,  7.5,      85.0},
        {120.0-lowest_IQ,  9.0,      94.0},
        {95.0-lowest_IQ,   3.5,      70.0},
        {110.0-lowest_IQ,  5.0,      88.0},
        {130.0-lowest_IQ,  8.0,      96.0},
        {115.0-lowest_IQ,  6.5,      87.0},
        {98.0-lowest_IQ,   4.0,      72.0},
        {125.0-lowest_IQ,  7.0,      91.0},
        {100.0-lowest_IQ,  3.0,      68.0},
        {118.0-lowest_IQ,  8.5,      89.0},
        {90.0-lowest_IQ,   2.5,      65.0},
        {135.0-lowest_IQ,  9.5,      98.0},
        {122.0-lowest_IQ,  7.8,      90.0},
        {88.0-lowest_IQ,   2.0,      60.0},
        {103.0-lowest_IQ,  5.5,      79.0},
        {112.0-lowest_IQ,  6.8,      84.0},
        {96.0-lowest_IQ,   3.8,      73.0},
        {116.0-lowest_IQ,  7.2,      88.0}
    };
    
    // Extract features (X) and target (y) from the data
    size_t n_samples = student_data.size();
    Matrix X(n_samples, 2);  // 2 features: IQ and Study Time
    Vector y(n_samples);     // Target: Grade
    
    for (size_t i = 0; i < n_samples; ++i) {
        X[i][0] = student_data[i][0];  // IQ
        X[i][1] = student_data[i][1];  // Study Time
        y[i] = student_data[i][2];     // Grade
    }
    
    // Extract feature columns
    Vector iq_values = X.get_col(0);
    Vector study_times = X.get_col(1);
    
    // Basic statistics
    std::cout << "\n===== Basic Statistics =====" << std::endl;
    std::cout << "Number of students: " << n_samples << std::endl;
    
    std::cout << "\nIQ Statistics:" << std::endl;
    std::cout << "  Mean: " << iq_values.mean() + lowest_IQ<< std::endl;
    std::cout << "  Standard Deviation: " << iq_values.std_dev() << std::endl;
    
    std::cout << "\nStudy Time Statistics (hours/day):" << std::endl;
    std::cout << "  Mean: " << study_times.mean() << std::endl;
    std::cout << "  Standard Deviation: " << study_times.std_dev() << std::endl;
    
    std::cout << "\nGrade Statistics:" << std::endl;
    std::cout << "  Mean: " << y.mean() << std::endl;
    std::cout << "  Standard Deviation: " << y.std_dev() << std::endl;
    
    // Correlation analysis
    std::cout << "\n===== Correlation Analysis =====" << std::endl;
    double corr_iq_grade = iq_values.correlation(y);
    double corr_study_grade = study_times.correlation(y);
    double corr_iq_study = iq_values.correlation(study_times);
    
    std::cout << "Correlation between IQ and Grades: " << corr_iq_grade << std::endl;
    std::cout << "Correlation between Study Time and Grades: " << corr_study_grade << std::endl;
    std::cout << "Correlation between IQ and Study Time: " << corr_iq_study << std::endl;
    
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
    
    // Model 1: Grade = f(IQ)
    std::cout << "\nModel 1: Grade as a function of IQ" << std::endl;
    
    Matrix X_iq_train(X_train.num_rows(), 1);
    Matrix X_iq_test(X_test.num_rows(), 1);
    
    for (size_t i = 0; i < X_train.num_rows(); ++i) {
        X_iq_train[i][0] = X_train[i][0];  // IQ
    }
    
    for (size_t i = 0; i < X_test.num_rows(); ++i) {
        X_iq_test[i][0] = X_test[i][0];  // IQ
    }
    
    try {
        // Use a smaller learning rate and enable feature scaling
        LinearRegression model1(0.05, 20000, 1e-6, false, true);
        model1.fit(X_iq_train, y_train);
        
        Vector y_pred1 = model1.predict(X_iq_test);
        double mse1 = mean_squared_error(y_test, y_pred1);
        double r2_1 = model1.r_squared(X_iq_test, y_test);
        
        std::cout << "  Coefficient (IQ): " << model1.get_weights()[0] << std::endl;
        std::cout << "  Intercept: " << model1.get_bias() << std::endl;
        std::cout << "  Mean Squared Error: " << mse1 << std::endl;
        std::cout << "  R-squared: " << r2_1 << std::endl;
        std::cout << "  Formula: Grade = " << model1.get_weights()[0] << " * IQ + " << model1.get_bias() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 1: " << e.what() << std::endl;
    }
    
    // Model 2: Grade = f(Study Time)
    std::cout << "\nModel 2: Grade as a function of Study Time" << std::endl;
    
    Matrix X_study_train(X_train.num_rows(), 1);
    Matrix X_study_test(X_test.num_rows(), 1);
    
    for (size_t i = 0; i < X_train.num_rows(); ++i) {
        X_study_train[i][0] = X_train[i][1];  // Study Time
    }
    
    for (size_t i = 0; i < X_test.num_rows(); ++i) {
        X_study_test[i][0] = X_test[i][1];  // Study Time
    }
    
    try {
        LinearRegression model2(0.05, 20000, 1e-6, false, true);
        model2.fit(X_study_train, y_train);
        
        Vector y_pred2 = model2.predict(X_study_test);
        double mse2 = mean_squared_error(y_test, y_pred2);
        double r2_2 = model2.r_squared(X_study_test, y_test);
        
        std::cout << "  Coefficient (Study Time): " << model2.get_weights()[0] << std::endl;
        std::cout << "  Intercept: " << model2.get_bias() << std::endl;
        std::cout << "  Mean Squared Error: " << mse2 << std::endl;
        std::cout << "  R-squared: " << r2_2 << std::endl;
        std::cout << "  Formula: Grade = " << model2.get_weights()[0] << " * StudyTime + " << model2.get_bias() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 2: " << e.what() << std::endl;
    }
    
    // Model 3: Grade = f(IQ, Study Time)
    std::cout << "\nModel 3: Grade as a function of both IQ and Study Time" << std::endl;
    
    try {
        LinearRegression model3(0.05, 20000, 1e-6, false, true);
        model3.fit(X_train, y_train);
        
        Vector y_pred3 = model3.predict(X_test);
        double mse3 = mean_squared_error(y_test, y_pred3);
        double r2_3 = model3.r_squared(X_test, y_test);
        
        Vector coeffs = model3.get_weights();
        std::cout << "  Coefficient (IQ): " << coeffs[0] << std::endl;
        std::cout << "  Coefficient (Study Time): " << coeffs[1] << std::endl;
        std::cout << "  Intercept: " << model3.get_bias() << std::endl;
        std::cout << "  Mean Squared Error: " << mse3 << std::endl;
        std::cout << "  R-squared: " << r2_3 << std::endl;
        std::cout << "  Formula: Grade = " << coeffs[0] << " * IQ + " 
                  << coeffs[1] << " * StudyTime + " << model3.get_bias() << std::endl;
        
        // Analysis of model performance
        std::cout << "\n===== Model Comparison =====" << std::endl;
        std::cout << "Model 1 (IQ only) - R-squared: ";
        try {
            LinearRegression model1(0.05, 20000, 1e-6, false, true);
            model1.fit(X_iq_train, y_train);
            std::cout << model1.r_squared(X_iq_test, y_test) << std::endl;
        } catch(...) {
            std::cout << "Could not compute" << std::endl;
        }
        
        std::cout << "Model 2 (Study Time only) - R-squared: ";
        try {
            LinearRegression model2(0.05, 20000, 1e-6, false, true);
            model2.fit(X_study_train, y_train);
            std::cout << model2.r_squared(X_study_test, y_test) << std::endl;
        } catch(...) {
            std::cout << "Could not compute" << std::endl;
        }
        
        std::cout << "Model 3 (IQ and Study Time) - R-squared: " << r2_3 << std::endl;
        
        // Standardized coefficients to compare feature importance
        Matrix X_std = standardize(X);
        MultipleLinearRegression std_model(0.05, 20000, 1e-6, false, true);
        std_model.fit(X_std, y);
        
        Vector std_coeffs = std_model.get_weights();
        std::cout << "\n===== Feature Importance Analysis =====" << std::endl;
        std::cout << "Standardized Coefficients:" << std::endl;
        std::cout << "  IQ: " << std_coeffs[0] << std::endl;
        std::cout << "  Study Time: " << std_coeffs[1] << std::endl;
        
        // Calculate predicted grades for different IQ and study time combinations
        std::cout << "\n===== Grade Predictions for Different Student Profiles =====" << std::endl;
        
        struct StudentProfile {
            double iq;
            double study_time;
            std::string description;
        };
        
        std::vector<StudentProfile> profiles = {
            {90.0-lowest_IQ, 2.0, "Low IQ, Low Study Time"},
            {90.0-lowest_IQ, 8.0, "Low IQ, High Study Time"},
            {110.0-lowest_IQ, 5.0, "Average IQ, Average Study Time"},
            {130.0-lowest_IQ, 2.0, "High IQ, Low Study Time"},
            {130.0-lowest_IQ, 8.0, "High IQ, High Study Time"}
        };
        
        Matrix profile_matrix(profiles.size(), 2);
        for (size_t i = 0; i < profiles.size(); ++i) {
            profile_matrix[i][0] = profiles[i].iq;
            profile_matrix[i][1] = profiles[i].study_time;
        }
        
        Vector predictions = model3.predict(profile_matrix);
        
        std::cout << std::setw(30) << std::left << "Student Profile" 
                  << std::setw(10) << std::left << "IQ" 
                  << std::setw(15) << std::left << "Study Time (h)" 
                  << std::setw(10) << std::left << "Grade" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (size_t i = 0; i < profiles.size(); ++i) {
            std::cout << std::setw(30) << std::left << profiles[i].description 
                      << std::setw(10) << std::left << profiles[i].iq 
                      << std::setw(15) << std::left << profiles[i].study_time 
                      << std::setw(10) << std::left << predictions[i] << std::endl;
        }
        double coeff_iq = coeffs[0];
        double coeff_study = coeffs[1];
        double intercept = model3.get_bias();
        
       
        
        // Study time effect at different IQ levels
        std::cout << "\n===== Effect of Increasing Study Time by 1 Hour =====" << std::endl;
        
        double study_time_coeff = coeffs[1];
        std::cout << "Grade increase per additional hour of study: " << study_time_coeff << " points" << std::endl;
        
        // Calculate "IQ equivalent" of study time
        std::cout << "\n===== 'IQ Equivalent' of Study Time =====" << std::endl;
        
        if (std::abs(coeff_iq) < 1e-10 || std::abs(coeff_study) < 1e-10) {
            std::cout << "Cannot calculate IQ equivalent (one or both coefficients are too small)" << std::endl;
        } else {
            double hours_per_iq_point = coeff_iq / coeff_study;
            
            std::cout << "One IQ point is equivalent to " << hours_per_iq_point << " hours of study time" << std::endl;
            std::cout << "5 hours of additional study time is equivalent to " << (5.0 * coeff_study / coeff_iq) << " IQ points" << std::endl;
        }
        
        // Interactive Grade Prediction Tool
        std::cout << "\n===== Interactive Grade Prediction Tool =====" << std::endl;
        std::cout << "Enter IQ and study time to predict a student's grade." << std::endl;
        
        char continue_prediction = 'y';
        while (continue_prediction == 'y' || continue_prediction == 'Y') {
            double input_iq, input_study_time;
            
            std::cout << "\nEnter student's IQ: ";
            std::cin >> input_iq;
            
            std::cout << "Enter daily study time (hours): ";
            std::cin >> input_study_time;
            
            // Create a single-row matrix for the input
            Matrix input_data(1, 2);
            input_data[0][0] = input_iq - lowest_IQ;  // Apply the same IQ adjustment used in training
            input_data[0][1] = input_study_time;
            
            // Predict using the trained model
            Vector predicted_grade = model3.predict(input_data);
            
            std::cout << "\nStudent Profile:" << std::endl;
            std::cout << "  IQ: " << input_iq << std::endl;
            std::cout << "  Study Time: " << input_study_time << " hours/day" << std::endl;
            std::cout << "  Predicted Grade: " << predicted_grade[0] << std::endl;
            
            // Optional: Show contribution of each factor
            double iq_contribution = coeff_iq * (input_iq - lowest_IQ);
            double study_contribution = coeff_study * input_study_time;
            
            std::cout << "\nContribution to grade:" << std::endl;
            std::cout << "  From IQ: " << iq_contribution << " points" << std::endl;
            std::cout << "  From Study Time: " << study_contribution << " points" << std::endl;
            std::cout << "  Base value: " << intercept << " points" << std::endl;
            
            // Suggest improvement
            std::cout << "\nFor each additional hour of study, grade could improve by " 
                      << coeff_study << " points." << std::endl;
            
            std::cout << "\nPredict another grade? (y/n): ";
            std::cin >> continue_prediction;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 3: " << e.what() << std::endl;
    }
    
    return 0;
}

//https://claude.ai/chat/cc1104fe-952e-4899-90f8-c9bc8679fa56