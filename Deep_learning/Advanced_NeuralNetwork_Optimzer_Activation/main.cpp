#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <cassert>
#include <set>

// ----- Vector Class -----
class Vector {
private:
    std::vector<double> data;
public:
    Vector() : data() {}
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& vec) : data(vec) {}
    double& operator[](size_t index) { return data[index]; }
    const double& operator[](size_t index) const { return data[index]; }
    size_t size() const { return data.size(); }
    
    Vector operator+(const Vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Vectors must have the same size for addition");
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = data[i] + other[i];
        return result;
    }
    
    Vector operator-(const Vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Vectors must have the same size for subtraction");
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = data[i] - other[i];
        return result;
    }
    
    Vector operator*(double scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = data[i] * scalar;
        return result;
    }
    
    Vector element_wise_multiply(const Vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Vectors must have the same size for element-wise multiplication");
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = data[i] * other[i];
        return result;
    }
    
    Vector element_wise_divide(const Vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Vectors must have the same size for element-wise division");
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = (std::abs(other[i]) < 1e-10) ? 0.0 : data[i] / other[i];
        return result;
    }
    
    Vector sqrt() const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i)
            result[i] = std::sqrt(std::max(0.0, data[i]));
        return result;
    }
    
    double dot(const Vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Vectors must have the same size for dot product");
        double result = 0.0;
        for (size_t i = 0; i < size(); ++i)
            result += data[i] * other[i];
        return result;
    }
    
    double sum() const {
        double result = 0.0;
        for (const auto& val : data)
            result += val;
        return result;
    }
    
    double mean() const {
        return (size() == 0) ? 0.0 : sum() / size();
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
    
    double correlation(const Vector& other) const {
        if (size() != other.size() || size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double mean_x = mean(), mean_y = other.mean();
        double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            double x_diff = data[i] - mean_x;
            double y_diff = other[i] - mean_y;
            sum_xy += x_diff * y_diff;
            sum_x2 += x_diff * x_diff;
            sum_y2 += y_diff * y_diff;
        }
        if (sum_x2 == 0.0 || sum_y2 == 0.0)
            return 0.0;
        return sum_xy / std::sqrt(sum_x2 * sum_y2);
    }
    
    const std::vector<double>& get_data() const { return data; }
    double max() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute max of empty vector");
        return *std::max_element(data.begin(), data.end());
    }
    
    double min() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute min of empty vector");
        return *std::min_element(data.begin(), data.end());
    }
    
    size_t argmax() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute argmax of empty vector");
        return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    }
    
    size_t argmin() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute argmin of empty vector");
        return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
    }
    
    bool has_nan() const {
        for (const auto& val : data)
            if (std::isnan(val))
                return true;
        return false;
    }
    
    void clip(double min_val, double max_val) {
        for (auto& val : data)
            val = std::max(min_val, std::min(max_val, val));
    }
};

// ----- Matrix Class -----
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
public:
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
    
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix dimensions must match for addition");
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i][j] = data[i][j] + other[i][j];
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i][j] = data[i][j] - other[i][j];
        return result;
    }
    
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i][j] = data[i][j] * scalar;
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        Matrix result(rows, other.cols, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < other.cols; ++j)
                for (size_t k = 0; k < cols; ++k)
                    result[i][j] += data[i][k] * other[k][j];
        return result;
    }
    
    Vector operator*(const Vector& vec) const {
        if (cols != vec.size())
            throw std::invalid_argument("Matrix-vector dimensions incompatible for multiplication");
        Vector result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i] += data[i][j] * vec[j];
        return result;
    }
    
    Vector get_col(size_t col) const {
        if (col >= cols)
            throw std::out_of_range("Column index out of range");
        Vector result(rows);
        for (size_t i = 0; i < rows; ++i)
            result[i] = data[i][col];
        return result;
    }
    
    Vector get_row(size_t row) const {
        if (row >= rows)
            throw std::out_of_range("Row index out of range");
        return Vector(data[row]);
    }
    
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[j][i] = data[i][j];
        return result;
    }
    
    bool has_nan() const {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                if (std::isnan(data[i][j]))
                    return true;
        return false;
    }
};

// ----- Activation Functions -----
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double compute(double x) const = 0;
    virtual Vector compute(const Vector& x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual Vector derivative(const Vector& x) const = 0;
    virtual std::string name() const = 0;
};

class Sigmoid : public ActivationFunction {
public:
    double compute(double x) const override {
        x = std::max(-500.0, std::min(500.0, x));
        return 1.0 / (1.0 + std::exp(-x));
    }
    Vector compute(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = compute(x[i]);
        return result;
    }
    double derivative(double x) const override {
        double s = compute(x);
        return s * (1.0 - s);
    }
    Vector derivative(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = derivative(x[i]);
        return result;
    }
    std::string name() const override { return "sigmoid"; }
};

class ReLU : public ActivationFunction {
public:
    double compute(double x) const override { return std::max(0.0, x); }
    Vector compute(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = compute(x[i]);
        return result;
    }
    double derivative(double x) const override { return x > 0.0 ? 1.0 : 0.0; }
    Vector derivative(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = derivative(x[i]);
        return result;
    }
    std::string name() const override { return "relu"; }
};

class LeakyReLU : public ActivationFunction {
private:
    double alpha;
public:
    LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
    double compute(double x) const override { return x > 0.0 ? x : alpha * x; }
    Vector compute(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = compute(x[i]);
        return result;
    }
    double derivative(double x) const override { return x > 0.0 ? 1.0 : alpha; }
    Vector derivative(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = derivative(x[i]);
        return result;
    }
    std::string name() const override { return "leaky_relu"; }
};

class Tanh : public ActivationFunction {
public:
    double compute(double x) const override {
        x = std::max(-500.0, std::min(500.0, x));
        return std::tanh(x);
    }
    Vector compute(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = compute(x[i]);
        return result;
    }
    double derivative(double x) const override {
        double t = compute(x);
        return 1.0 - t * t;
    }
    Vector derivative(const Vector& x) const override {
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = derivative(x[i]);
        return result;
    }
    std::string name() const override { return "tanh"; }
};

class Softmax : public ActivationFunction {
public:
    double compute(double x) const override {
        throw std::invalid_argument("Softmax not defined for a single value");
    }
    Vector compute(const Vector& x) const override {
        double max_val = x.max();
        Vector exp_values(x.size());
        double sum_exp = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double val = std::max(-500.0, std::min(500.0, x[i] - max_val));
            exp_values[i] = std::exp(val);
            sum_exp += exp_values[i];
        }
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = (sum_exp > 0.0) ? exp_values[i] / sum_exp : 1.0 / x.size();
        return result;
    }
    double derivative(double x) const override {
        throw std::invalid_argument("Softmax derivative not defined for a single value");
    }
    Vector derivative(const Vector& x) const override {
        throw std::invalid_argument("Softmax derivative requires a matrix and is typically handled in the loss function");
    }
    std::string name() const override { return "softmax"; }
};

class Identity : public ActivationFunction {
public:
    double compute(double x) const override { return x; }
    Vector compute(const Vector& x) const override { return x; }
    double derivative(double x) const override { return 1.0; }
    Vector derivative(const Vector& x) const override { return Vector(x.size(), 1.0); }
    std::string name() const override { return "identity"; }
};

// ----- Loss Functions -----
class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double compute(const Vector& y_true, const Vector& y_pred) const = 0;
    virtual Vector gradient(const Vector& y_true, const Vector& y_pred) const = 0;
    virtual std::string name() const = 0;
};

class MSELoss : public LossFunction {
public:
    double compute(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double diff = y_true[i] - y_pred[i];
            sum += diff * diff;
        }
        return sum / y_true.size();
    }
    Vector gradient(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector result(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i)
            result[i] = -2.0 * (y_true[i] - y_pred[i]) / y_true.size();
        return result;
    }
    std::string name() const override { return "mse"; }
};

class MAELoss : public LossFunction {
public:
    double compute(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i)
            sum += std::abs(y_true[i] - y_pred[i]);
        return sum / y_true.size();
    }
    Vector gradient(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector result(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            double diff = y_pred[i] - y_true[i];
            result[i] = (diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0)) / y_true.size();
        }
        return result;
    }
    std::string name() const override { return "mae"; }
};

class BinaryCrossEntropyLoss : public LossFunction {
private:
    const double epsilon = 1e-10;
public:
    double compute(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = std::max(std::min(y_pred[i], 1.0 - epsilon), epsilon);
            sum += y_true[i] * std::log(p) + (1.0 - y_true[i]) * std::log(1.0 - p);
        }
        return -sum / y_true.size();
    }
    Vector gradient(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector result(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = std::max(std::min(y_pred[i], 1.0 - epsilon), epsilon);
            result[i] = -(y_true[i] / p - (1.0 - y_true[i]) / (1.0 - p)) / y_true.size();
        }
        return result;
    }
    std::string name() const override { return "binary_crossentropy"; }
};

class CategoricalCrossEntropyLoss : public LossFunction {
private:
    const double epsilon = 1e-10;
public:
    double compute(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] > 0.0) {
                double p = std::max(y_pred[i], epsilon);
                sum += y_true[i] * std::log(p);
            }
        }
        return -sum;
    }
    Vector gradient(const Vector& y_true, const Vector& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector result(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i)
            result[i] = y_pred[i] - y_true[i];
        return result;
    }
    std::string name() const override { return "categorical_crossentropy"; }
};

// ----- Optimizers -----
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Vector& weights, const Vector& gradients) = 0;
    virtual void update_bias(double& bias, double gradient) = 0;
    virtual void reset() = 0;
    virtual std::string name() const = 0;
    
    void clip_gradient(Vector& gradients, double max_norm) {
        double norm = 0.0;
        for (size_t i = 0; i < gradients.size(); ++i)
            norm += gradients[i] * gradients[i];
        norm = std::sqrt(norm);
        if (norm > max_norm) {
            double scale = max_norm / norm;
            for (size_t i = 0; i < gradients.size(); ++i)
                gradients[i] *= scale;
        }
    }
};

class SGDMomentum : public Optimizer {
private:
    double learning_rate;
    double momentum;
    Vector velocity;
    double bias_velocity;
    double max_grad_norm;
public:
    SGDMomentum(double learning_rate = 0.01, double momentum = 0.9, double max_grad_norm = 1.0)
        : learning_rate(learning_rate), momentum(momentum), bias_velocity(0.0), max_grad_norm(max_grad_norm) {}
    
    void update(Vector& weights, const Vector& gradients) override {
        if (velocity.size() != weights.size())
            velocity = Vector(weights.size(), 0.0);
        Vector clipped_gradients = gradients;
        if (max_grad_norm > 0.0)
            clip_gradient(clipped_gradients, max_grad_norm);
        for (size_t i = 0; i < weights.size(); ++i) {
            velocity[i] = momentum * velocity[i] + learning_rate * clipped_gradients[i];
            weights[i] -= velocity[i];
        }
    }
    
    void update_bias(double& bias, double gradient) override {
        gradient = std::max(-max_grad_norm, std::min(max_grad_norm, gradient));
        bias_velocity = momentum * bias_velocity + learning_rate * gradient;
        bias -= bias_velocity;
    }
    
    void reset() override {
        velocity = Vector();
        bias_velocity = 0.0;
    }
    
    std::string name() const override { return "sgd_momentum"; }
};

class Adam : public Optimizer {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    Vector m;
    Vector v;
    double m_bias;
    double v_bias;
    int t;
    double max_grad_norm;
public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
         double epsilon = 1e-8, double max_grad_norm = 1.0)
         : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon),
           m_bias(0.0), v_bias(0.0), t(0), max_grad_norm(max_grad_norm) {}
    
    void update(Vector& weights, const Vector& gradients) override {
        if (m.size() != weights.size()) {
            m = Vector(weights.size(), 0.0);
            v = Vector(weights.size(), 0.0);
        }
        t++;
        Vector clipped_gradients = gradients;
        if (max_grad_norm > 0.0)
            clip_gradient(clipped_gradients, max_grad_norm);
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * clipped_gradients[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * clipped_gradients[i] * clipped_gradients[i];
            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));
            weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
    
    void update_bias(double& bias, double gradient) override {
        gradient = std::max(-max_grad_norm, std::min(max_grad_norm, gradient));
        m_bias = beta1 * m_bias + (1.0 - beta1) * gradient;
        v_bias = beta2 * v_bias + (1.0 - beta2) * gradient * gradient;
        double m_hat = m_bias / (1.0 - std::pow(beta1, t));
        double v_hat = v_bias / (1.0 - std::pow(beta2, t));
        bias -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    
    void reset() override {
        m = Vector();
        v = Vector();
        m_bias = 0.0;
        v_bias = 0.0;
        t = 0;
    }
    
    std::string name() const override { return "adam"; }
};

class RMSProp : public Optimizer {
private:
    double learning_rate;
    double decay_rate;
    double epsilon;
    Vector cache;
    double bias_cache;
    double max_grad_norm;
public:
    RMSProp(double learning_rate = 0.01, double decay_rate = 0.9,
            double epsilon = 1e-8, double max_grad_norm = 1.0)
            : learning_rate(learning_rate), decay_rate(decay_rate), epsilon(epsilon),
              bias_cache(0.0), max_grad_norm(max_grad_norm) {}
    
    void update(Vector& weights, const Vector& gradients) override {
        if (cache.size() != weights.size())
            cache = Vector(weights.size(), 0.0);
        Vector clipped_gradients = gradients;
        if (max_grad_norm > 0.0)
            clip_gradient(clipped_gradients, max_grad_norm);
        for (size_t i = 0; i < weights.size(); ++i) {
            cache[i] = decay_rate * cache[i] + (1.0 - decay_rate) * clipped_gradients[i] * clipped_gradients[i];
            weights[i] -= learning_rate * clipped_gradients[i] / (std::sqrt(cache[i]) + epsilon);
        }
    }
    
    void update_bias(double& bias, double gradient) override {
        gradient = std::max(-max_grad_norm, std::min(max_grad_norm, gradient));
        bias_cache = decay_rate * bias_cache + (1.0 - decay_rate) * gradient * gradient;
        bias -= learning_rate * gradient / (std::sqrt(bias_cache) + epsilon);
    }
    
    void reset() override {
        cache = Vector();
        bias_cache = 0.0;
    }
    
    std::string name() const override { return "rmsprop"; }
};

// ----- Preprocessing Tools -----
std::pair<Matrix, std::vector<std::pair<double, double>>> scale_features(const Matrix& X) {
    size_t n_samples = X.num_rows(), n_features = X.num_cols();
    std::vector<std::pair<double, double>> min_max(n_features);
    for (size_t j = 0; j < n_features; ++j) {
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < n_samples; ++i) {
            min_val = std::min(min_val, X[i][j]);
            max_val = std::max(max_val, X[i][j]);
        }
        min_max[j] = {min_val, max_val};
    }
    Matrix X_scaled(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j) {
            double range = min_max[j].second - min_max[j].first;
            X_scaled[i][j] = (range > 0) ? (X[i][j] - min_max[j].first) / range : 0.5;
        }
    return {X_scaled, min_max};
}

double unscale_value(double scaled_value, double min_val, double max_val) {
    double range = max_val - min_val;
    return (range > 0) ? scaled_value * range + min_val : min_val;
}

double scale_value(double value, double min_val, double max_val) {
    double range = max_val - min_val;
    return (range > 0) ? (value - min_val) / range : 0.5;
}

std::pair<Matrix, std::vector<std::pair<double, double>>> standardize_features(const Matrix& X) {
    size_t n_samples = X.num_rows(), n_features = X.num_cols();
    std::vector<std::pair<double, double>> mean_std(n_features);
    for (size_t j = 0; j < n_features; ++j) {
        Vector feature = X.get_col(j);
        double mean = feature.mean();
        double std_dev = feature.std_dev();
        mean_std[j] = {mean, std_dev};
    }
    Matrix X_std(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            X_std[i][j] = (mean_std[j].second > 0) ? (X[i][j] - mean_std[j].first) / mean_std[j].second : 0.0;
    return {X_std, mean_std};
}

double unstandardize_value(double std_value, double mean, double std_dev) {
    return (std_dev > 0) ? std_value * std_dev + mean : mean;
}

Matrix one_hot_encode(const Vector& categorical_feature, size_t num_categories = 0) {
    if (num_categories == 0) {
        std::set<double> unique_values;
        for (size_t i = 0; i < categorical_feature.size(); ++i)
            unique_values.insert(categorical_feature[i]);
        num_categories = unique_values.size();
    }
    std::unordered_map<double, size_t> value_to_index;
    size_t next_index = 0;
    for (size_t i = 0; i < categorical_feature.size(); ++i) {
        if (value_to_index.find(categorical_feature[i]) == value_to_index.end()) {
            value_to_index[categorical_feature[i]] = next_index++;
            if (next_index > num_categories)
                throw std::invalid_argument("More unique values than specified number of categories");
        }
    }
    Matrix encoded(categorical_feature.size(), num_categories, 0.0);
    for (size_t i = 0; i < categorical_feature.size(); ++i) {
        size_t index = value_to_index[categorical_feature[i]];
        encoded[i][index] = 1.0;
    }
    return encoded;
}

std::tuple<Matrix, Matrix, Vector, Vector> train_test_split(const Matrix& X, const Vector& y,
                                                             double test_size = 0.2, unsigned int random_seed = 42) {
    if (X.num_rows() != y.size() || X.num_rows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    size_t n_samples = X.num_rows();
    size_t n_test = static_cast<size_t>(n_samples * test_size);
    n_test = std::max(size_t(1), n_test);
    size_t n_train = n_samples - n_test;
    n_train = std::max(size_t(1), n_train);
    n_test = n_samples - n_train;
    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        indices[i] = i;
    std::mt19937 g(random_seed);
    std::shuffle(indices.begin(), indices.end(), g);
    Matrix X_train(n_train, X.num_cols());
    Matrix X_test(n_test, X.num_cols());
    Vector y_train(n_train);
    Vector y_test(n_test);
    for (size_t i = 0; i < n_train; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.num_cols(); ++j)
            X_train[i][j] = X[idx][j];
        y_train[i] = y[idx];
    }
    for (size_t i = 0; i < n_test; ++i) {
        size_t idx = indices[i + n_train];
        for (size_t j = 0; j < X.num_cols(); ++j)
            X_test[i][j] = X[idx][j];
        y_test[i] = y[idx];
    }
    return {X_train, X_test, y_train, y_test};
}

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> k_fold_indices(size_t n_samples, size_t k, unsigned int random_seed = 42) {
    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        indices[i] = i;
    std::mt19937 g(random_seed);
    std::shuffle(indices.begin(), indices.end(), g);
    std::vector<size_t> fold_sizes(k, n_samples / k);
    for (size_t i = 0; i < n_samples % k; ++i)
        fold_sizes[i]++;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> splits;
    size_t start = 0;
    for (size_t i = 0; i < k; ++i) {
        size_t fold_size = fold_sizes[i];
        std::vector<size_t> val_indices(indices.begin() + start, indices.begin() + start + fold_size);
        std::vector<size_t> train_indices;
        for (size_t j = 0; j < n_samples; ++j)
            if (j < start || j >= start + fold_size)
                train_indices.push_back(indices[j]);
        splits.push_back({train_indices, val_indices});
        start += fold_size;
    }
    return splits;
}

template<typename ModelType>
std::vector<double> cross_validate(const Matrix& X, const Vector& y, size_t k,
                                   const std::function<ModelType()>& model_factory,
                                   const std::function<double(const ModelType&, const Matrix&, const Vector&)>& score_func,
                                   unsigned int random_seed = 42) {
    if (X.num_rows() != y.size() || X.num_rows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    size_t n_samples = X.num_rows();
    auto fold_indices = k_fold_indices(n_samples, k, random_seed);
    std::vector<double> scores;
    for (size_t i = 0; i < k; ++i) {
        const auto& train_indices = fold_indices[i].first;
        const auto& val_indices = fold_indices[i].second;
        Matrix X_train(train_indices.size(), X.num_cols());
        Vector y_train(train_indices.size());
        Matrix X_val(val_indices.size(), X.num_cols());
        Vector y_val(val_indices.size());
        for (size_t j = 0; j < train_indices.size(); ++j) {
            size_t idx = train_indices[j];
            for (size_t c = 0; c < X.num_cols(); ++c)
                X_train[j][c] = X[idx][c];
            y_train[j] = y[idx];
        }
        for (size_t j = 0; j < val_indices.size(); ++j) {
            size_t idx = val_indices[j];
            for (size_t c = 0; c < X.num_cols(); ++c)
                X_val[j][c] = X[idx][c];
            y_val[j] = y[idx];
        }
        ModelType model = model_factory();
        model.fit(X_train, y_train);
        double score = score_func(model, X_val, y_val);
        scores.push_back(score);
    }
    return scores;
}

template<typename ModelType, typename ParamType>
std::pair<ParamType, double> grid_search_cv(const Matrix& X, const Vector& y,
                                            const std::vector<ParamType>& param_grid,
                                            const std::function<ModelType(const ParamType&)>& model_factory,
                                            const std::function<double(const ModelType&, const Matrix&, const Vector&)>& score_func,
                                            size_t k = 5, unsigned int random_seed = 42) {
    if (X.num_rows() != y.size() || X.num_rows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    if (param_grid.empty())
        throw std::invalid_argument("Parameter grid cannot be empty");
    double best_score = -std::numeric_limits<double>::max();
    ParamType best_params = param_grid[0];
    for (const auto& params : param_grid) {
        auto fixed_model_factory = [&]() { return model_factory(params); };
        auto scores = cross_validate<ModelType>(X, y, k, fixed_model_factory, score_func, random_seed);
        double mean_score = 0.0;
        for (const auto& score : scores)
            mean_score += score;
        mean_score /= scores.size();
        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }
    return {best_params, best_score};
}

// ----- Metrics -----
double mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / y_true.size();
}

double mean_absolute_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        sum += std::abs(y_true[i] - y_pred[i]);
    return sum / y_true.size();
}

double r_squared(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double y_mean = y_true.mean();
    double ss_total = 0.0, ss_residual = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff_total = y_true[i] - y_mean;
        double diff_residual = y_true[i] - y_pred[i];
        ss_total += diff_total * diff_total;
        ss_residual += diff_residual * diff_residual;
    }
    return (ss_total < 1e-10) ? 0.0 : 1.0 - (ss_residual / ss_total);
}

double accuracy(const Vector& y_true, const Vector& y_pred, double threshold = 0.5) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        bool true_class = y_true[i] > threshold;
        bool pred_class = y_pred[i] > threshold;
        if (true_class == pred_class)
            correct++;
    }
    return static_cast<double>(correct) / y_true.size();
}

// ----- Neural Network Layer -----
class Layer {
public:
    virtual ~Layer() = default;
    virtual Vector forward(const Vector& input) = 0;
    virtual Vector backward(const Vector& grad_output) = 0;
    virtual void update(Optimizer& optimizer) = 0;
    virtual void reset() = 0;
};

class DenseLayer : public Layer {
private:
    size_t input_size;
    size_t output_size;
    Matrix weights;
    Vector biases;
    Vector input;
    std::shared_ptr<ActivationFunction> activation;
    Vector output_before_activation;
    Vector weights_gradient;
public:
    DenseLayer(size_t input_size, size_t output_size, 
               std::shared_ptr<ActivationFunction> activation = std::make_shared<Identity>())
        : input_size(input_size), output_size(output_size),
          weights(output_size, input_size), biases(output_size, 0.0),
          activation(activation), weights_gradient(output_size * input_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<double> dis(-limit, limit);
        for (size_t i = 0; i < output_size; ++i)
            for (size_t j = 0; j < input_size; ++j)
                weights[i][j] = dis(gen) * 0.1;
    }
    
    Vector forward(const Vector& input) override {
        if (input.has_nan())
            throw std::runtime_error("NaN detected in input to dense layer");
        this->input = input;
        output_before_activation = Vector(output_size, 0.0);
        for (size_t i = 0; i < output_size; ++i) {
            double sum = biases[i];
            for (size_t j = 0; j < input_size; ++j)
                sum += weights[i][j] * input[j];
            output_before_activation[i] = sum;
        }
        Vector activated = activation->compute(output_before_activation);
        if (activated.has_nan())
            throw std::runtime_error("NaN detected in dense layer output after activation");
        return activated;
    }
    
    Vector backward(const Vector& grad_output) override {
        if (grad_output.has_nan())
            throw std::runtime_error("NaN detected in gradient output");
        Vector grad_output_before_activation = grad_output;
        if (activation->name() != "identity") {
            Vector activation_derivative = activation->derivative(output_before_activation);
            for (size_t i = 0; i < output_size; ++i)
                grad_output_before_activation[i] *= activation_derivative[i];
        }
        Vector grad_input(input_size, 0.0);
        for (size_t i = 0; i < input_size; ++i)
            for (size_t j = 0; j < output_size; ++j)
                grad_input[i] += weights[j][i] * grad_output_before_activation[j];
        size_t idx = 0;
        for (size_t i = 0; i < output_size; ++i)
            for (size_t j = 0; j < input_size; ++j)
                weights_gradient[idx++] = grad_output_before_activation[i] * input[j];
        for (size_t i = 0; i < output_size; ++i)
            biases[i] = grad_output_before_activation[i];
        if (grad_input.has_nan())
            throw std::runtime_error("NaN detected in gradient input");
        return grad_input;
    }
    
    void update(Optimizer& optimizer) override {
        Vector flat_weights(output_size * input_size);
        size_t idx = 0;
        for (size_t i = 0; i < output_size; ++i)
            for (size_t j = 0; j < input_size; ++j)
                flat_weights[idx++] = weights[i][j];
        if (flat_weights.has_nan() || weights_gradient.has_nan())
            throw std::runtime_error("NaN detected in weights or gradients");
        optimizer.update(flat_weights, weights_gradient);
        idx = 0;
        for (size_t i = 0; i < output_size; ++i)
            for (size_t j = 0; j < input_size; ++j)
                weights[i][j] = flat_weights[idx++];
        Vector bias_gradients = biases;
        for (size_t i = 0; i < output_size; ++i)
            optimizer.update_bias(biases[i], bias_gradients[i]);
    }
    
    void reset() override {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<double> dis(-limit, limit);
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j)
                weights[i][j] = dis(gen) * 0.1;
            biases[i] = 0.0;
        }
    }
};

// ----- Neural Network Model -----
class NeuralNetwork {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<LossFunction> loss_function;
    std::shared_ptr<Optimizer> optimizer;
    bool verbose;
    double y_min;
    double y_max;
public:
    NeuralNetwork(std::shared_ptr<LossFunction> loss_function = std::make_shared<MSELoss>(),
                  std::shared_ptr<Optimizer> optimizer = std::make_shared<SGDMomentum>(),
                  bool verbose = false)
        : loss_function(loss_function), optimizer(optimizer), verbose(verbose), y_min(0.0), y_max(1.0) {}
    
    void add_layer(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }
    
    Vector forward(const Vector& input) const {
        Vector output = input;
        for (const auto& layer : layers) {
            try {
                output = layer->forward(output);
            } catch (const std::exception& e) {
                std::cerr << "Error in forward pass: " << e.what() << std::endl;
                throw;
            }
        }
        return output;
    }
    
    void backward(const Vector& input, const Vector& target) {
        try {
            Vector output = forward(input);
            Vector grad_output = loss_function->gradient(target, output);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                grad_output = (*it)->backward(grad_output);
        } catch (const std::exception& e) {
            std::cerr << "Error in backward pass: " << e.what() << std::endl;
            throw;
        }
    }
    
    void update() {
        try {
            for (auto& layer : layers)
                layer->update(*optimizer);
        } catch (const std::exception& e) {
            std::cerr << "Error in update: " << e.what() << std::endl;
            throw;
        }
    }
    
    void reset() {
        for (auto& layer : layers)
            layer->reset();
        optimizer->reset();
    }
    
    void fit(const Matrix& X, const Vector& y, size_t batch_size = 32, size_t epochs = 100) {
        size_t n_samples = X.num_rows();
        if (n_samples != y.size())
            throw std::invalid_argument("Number of samples must match number of targets");
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        y_min = y.min();
        y_max = y.max();
        double y_range = y_max - y_min;
        Vector y_normalized(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
            y_normalized[i] = (y_range > 0) ? (y[i] - y_min) / y_range : 0.5;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            double total_loss = 0.0;
            for (size_t i = 0; i < n_samples; i += batch_size) {
                size_t batch_end = std::min(i + batch_size, n_samples);
                for (size_t j = i; j < batch_end; ++j) {
                    size_t idx = indices[j];
                    Vector input = X.get_row(idx);
                    Vector target(1, y_normalized[idx]);
                    try {
                        backward(input, target);
                        Vector output = forward(input);
                        total_loss += loss_function->compute(target, output);
                    } catch (const std::exception& e) {
                        std::cerr << "Error during training: " << e.what() << std::endl;
                        reset();
                        break;
                    }
                }
                try {
                    update();
                } catch (const std::exception& e) {
                    std::cerr << "Error updating parameters: " << e.what() << std::endl;
                    reset();
                    break;
                }
            }
            double avg_loss = total_loss / n_samples;
            if (verbose && (epoch % 10 == 0 || epoch == epochs - 1))
                std::cout << "Epoch " << epoch << ": loss = " << avg_loss << std::endl;
            if (avg_loss < 1e-5) {
                if (verbose)
                    std::cout << "Early stopping at epoch " << epoch << " with loss " << avg_loss << std::endl;
                break;
            }
            if (std::isnan(avg_loss) || std::isinf(avg_loss))
                reset();
        }
    }
    
    Vector predict(const Matrix& X) const {
        size_t n_samples = X.num_rows();
        Vector predictions(n_samples);
        try {
            for (size_t i = 0; i < n_samples; ++i) {
                Vector input = X.get_row(i);
                Vector output = forward(input);
                predictions[i] = (y_max > y_min) ? output[0] * (y_max - y_min) + y_min : output[0];
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
            throw;
        }
        return predictions;
    }
    
    double evaluate(const Matrix& X, const Vector& y) const {
        Vector predictions = predict(X);
        return loss_function->compute(y, predictions);
    }
    
    double r_squared(const Matrix& X, const Vector& y) const {
        Vector predictions = predict(X);
        double y_mean = y.mean();
        double ss_total = 0.0, ss_residual = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff_total = y[i] - y_mean;
            double diff_residual = y[i] - predictions[i];
            ss_total += diff_total * diff_total;
            ss_residual += diff_residual * diff_residual;
        }
        return (ss_total < 1e-10) ? 0.0 : 1.0 - (ss_residual / ss_total);
    }
};

class LinearRegression {
private:
    double learning_rate;
    int max_iterations;
    double tol;
    bool verbose;
    double bias;
    std::vector<std::pair<double, double>> feature_min_max;
    double target_min;
    double target_max;
    bool use_scaling;
    std::shared_ptr<Optimizer> optimizer;
    Vector weights;
public:
    LinearRegression(double learning_rate = 0.001, int max_iterations = 1000000, 
                     double tol = 1e-6, bool verbose = false, bool use_scaling = true,
                     std::shared_ptr<Optimizer> optimizer = std::make_shared<SGDMomentum>())
        : learning_rate(learning_rate), max_iterations(max_iterations), tol(tol), verbose(verbose),
          bias(0.0), target_min(0.0), target_max(1.0), use_scaling(use_scaling), optimizer(optimizer) {}
    
    void fit(const Matrix& X_orig, const Vector& y_orig) {
        if (X_orig.num_rows() != y_orig.size() || X_orig.num_rows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Matrix X;
        Vector y;
        if (use_scaling) {
            auto [X_scaled, min_max] = scale_features(X_orig);
            X = X_scaled;
            feature_min_max = min_max;
            target_min = *std::min_element(y_orig.get_data().begin(), y_orig.get_data().end());
            target_max = *std::max_element(y_orig.get_data().begin(), y_orig.get_data().end());
            double target_range = target_max - target_min;
            y = Vector(y_orig.size());
            for (size_t i = 0; i < y_orig.size(); ++i)
                y[i] = (target_range > 0) ? (y_orig[i] - target_min) / target_range : 0.5;
        } else {
            X = X_orig;
            y = y_orig;
        }
        size_t n_samples = X.num_rows();
        size_t n_features = X.num_cols();
        weights = Vector(n_features, 0.0);
        bias = 0.0;
        optimizer->reset();
        double prev_loss = std::numeric_limits<double>::max();
        for (int iter = 0; iter < max_iterations; ++iter) {
            Vector y_pred(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = bias;
                for (size_t j = 0; j < n_features; ++j)
                    pred += X[i][j] * weights[j];
                y_pred[i] = pred;
            }
            double loss = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y[i];
                loss += error * error;
            }
            loss /= n_samples;
            if (std::abs(loss - prev_loss) < tol) {
                if (verbose)
                    std::cout << "Converged at iteration " << iter << " with loss " << loss << std::endl;
                break;
            }
            prev_loss = loss;
            Vector grad_w(n_features, 0.0);
            double grad_b = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y[i];
                for (size_t j = 0; j < n_features; ++j)
                    grad_w[j] += error * X[i][j];
                grad_b += error;
            }
            for (size_t j = 0; j < n_features; ++j)
                grad_w[j] /= n_samples;
            grad_b /= n_samples;
            optimizer->update(weights, grad_w);
            optimizer->update_bias(bias, grad_b);
            if (verbose && (iter % 1000 == 0 || iter == max_iterations - 1)) {
                std::cout << "Iteration " << iter << ": loss = " << loss << std::endl;
                std::cout << "  weights = [";
                for (size_t j = 0; j < n_features; ++j) {
                    std::cout << weights[j];
                    if (j < n_features - 1)
                        std::cout << ", ";
                }
                std::cout << "], bias = " << bias << std::endl;
            }
        }
    }
    
    Vector predict(const Matrix& X_orig) const {
        if (X_orig.num_cols() != weights.size())
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        Matrix X;
        if (use_scaling && !feature_min_max.empty()) {
            size_t n_samples = X_orig.num_rows();
            size_t n_features = X_orig.num_cols();
            X = Matrix(n_samples, n_features);
            for (size_t i = 0; i < n_samples; ++i)
                for (size_t j = 0; j < n_features; ++j) {
                    double range = feature_min_max[j].second - feature_min_max[j].first;
                    X[i][j] = (range > 0) ? (X_orig[i][j] - feature_min_max[j].first) / range : 0.5;
                }
        } else {
            X = X_orig;
        }
        size_t n_samples = X.num_rows();
        Vector y_pred(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            double pred = bias;
            for (size_t j = 0; j < X.num_cols(); ++j)
                pred += X[i][j] * weights[j];
            y_pred[i] = pred;
        }
        if (use_scaling) {
            double target_range = target_max - target_min;
            for (size_t i = 0; i < n_samples; ++i)
                y_pred[i] = (target_range > 0) ? y_pred[i] * target_range + target_min : target_min;
        }
        return y_pred;
    }
    
    Vector get_weights() const {
        if (!use_scaling || feature_min_max.empty())
            return weights;
        Vector orig_weights(weights.size());
        double target_range = target_max - target_min;
        for (size_t j = 0; j < weights.size(); ++j) {
            double feature_range = feature_min_max[j].second - feature_min_max[j].first;
            orig_weights[j] = (feature_range > 0 && target_range > 0) ? weights[j] * target_range / feature_range : weights[j];
        }
        return orig_weights;
    }
    
    double get_bias() const {
        if (!use_scaling)
            return bias;
        double target_range = target_max - target_min;
        double unscaled_bias = bias * target_range + target_min;
        for (size_t j = 0; j < weights.size(); ++j) {
            double feature_min = feature_min_max[j].first;
            double feature_range = feature_min_max[j].second - feature_min;
            if (feature_range > 0)
                unscaled_bias -= weights[j] * target_range * feature_min / feature_range;
        }
        return unscaled_bias;
    }
    
    double r_squared(const Matrix& X, const Vector& y) const {
        if (X.num_rows() != y.size() || X.num_rows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Vector y_pred = predict(X);
        double y_mean = y.mean();
        double ss_total = 0.0, ss_residual = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff_total = y[i] - y_mean;
            double diff_residual = y[i] - y_pred[i];
            ss_total += diff_total * diff_total;
            ss_residual += diff_residual * diff_residual;
        }
        return (ss_total == 0.0) ? 0.0 : 1.0 - (ss_residual / ss_total);
    }
    
    std::shared_ptr<Optimizer> get_optimizer() const { return optimizer; }
    void set_optimizer(std::shared_ptr<Optimizer> new_optimizer) { optimizer = new_optimizer; }
};

typedef LinearRegression MultipleLinearRegression;

class LogisticRegression {
private:
    double learning_rate;
    int max_iterations;
    double tol;
    bool verbose;
    double bias;
    Vector weights;
    std::vector<std::pair<double, double>> feature_min_max;
    bool use_scaling;
    std::shared_ptr<Optimizer> optimizer;
    Sigmoid sigmoid;
public:
    LogisticRegression(double learning_rate = 0.001, int max_iterations = 1000000, 
                       double tol = 1e-6, bool verbose = false, bool use_scaling = true,
                       std::shared_ptr<Optimizer> optimizer = std::make_shared<SGDMomentum>())
        : learning_rate(learning_rate), max_iterations(max_iterations), tol(tol), verbose(verbose),
          bias(0.0), use_scaling(use_scaling), optimizer(optimizer) {}
    
    void fit(const Matrix& X_orig, const Vector& y_orig) {
        if (X_orig.num_rows() != y_orig.size() || X_orig.num_rows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Matrix X;
        if (use_scaling) {
            auto [X_scaled, min_max] = scale_features(X_orig);
            X = X_scaled;
            feature_min_max = min_max;
        } else {
            X = X_orig;
        }
        size_t n_samples = X.num_rows();
        size_t n_features = X.num_cols();
        weights = Vector(n_features, 0.0);
        bias = 0.0;
        optimizer->reset();
        double prev_loss = std::numeric_limits<double>::max();
        for (int iter = 0; iter < max_iterations; ++iter) {
            Vector z(n_samples);
            Vector y_pred(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = bias;
                for (size_t j = 0; j < n_features; ++j)
                    pred += X[i][j] * weights[j];
                z[i] = pred;
                y_pred[i] = sigmoid.compute(pred);
            }
            double loss = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double y = y_orig[i];
                double p = std::max(std::min(y_pred[i], 1.0 - 1e-15), 1e-15);
                loss += y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
            }
            loss = -loss / n_samples;
            if (std::abs(loss - prev_loss) < tol) {
                if (verbose)
                    std::cout << "Converged at iteration " << iter << " with loss " << loss << std::endl;
                break;
            }
            prev_loss = loss;
            Vector grad_w(n_features, 0.0);
            double grad_b = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y_orig[i];
                for (size_t j = 0; j < n_features; ++j)
                    grad_w[j] += error * X[i][j];
                grad_b += error;
            }
            for (size_t j = 0; j < n_features; ++j)
                grad_w[j] /= n_samples;
            grad_b /= n_samples;
            optimizer->update(weights, grad_w);
            optimizer->update_bias(bias, grad_b);
            if (verbose && (iter % 1000 == 0 || iter == max_iterations - 1))
                std::cout << "Iteration " << iter << ": loss = " << loss << std::endl;
        }
    }
    
    Vector predict_proba(const Matrix& X_orig) const {
        if (X_orig.num_cols() != weights.size())
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        Matrix X;
        if (use_scaling && !feature_min_max.empty()) {
            size_t n_samples = X_orig.num_rows();
            size_t n_features = X_orig.num_cols();
            X = Matrix(n_samples, n_features);
            for (size_t i = 0; i < n_samples; ++i)
                for (size_t j = 0; j < n_features; ++j) {
                    double range = feature_min_max[j].second - feature_min_max[j].first;
                    X[i][j] = (range > 0) ? (X_orig[i][j] - feature_min_max[j].first) / range : 0.5;
                }
        } else {
            X = X_orig;
        }
        size_t n_samples = X.num_rows();
        Vector probas(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            double z = bias;
            for (size_t j = 0; j < X.num_cols(); ++j)
                z += X[i][j] * weights[j];
            probas[i] = sigmoid.compute(z);
        }
        return probas;
    }
    
    Vector predict(const Matrix& X, double threshold = 0.5) const {
        Vector probas = predict_proba(X);
        size_t n_samples = probas.size();
        Vector predictions(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
            predictions[i] = (probas[i] >= threshold) ? 1.0 : 0.0;
        return predictions;
    }
    
    Vector get_weights() const { return weights; }
    double get_bias() const { return bias; }
    
    double accuracy(const Matrix& X, const Vector& y, double threshold = 0.5) const {
        if (X.num_rows() != y.size() || X.num_rows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Vector y_pred = predict(X, threshold);
        size_t correct = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if ((y[i] >= threshold && y_pred[i] >= threshold) || (y[i] < threshold && y_pred[i] < threshold))
                correct++;
        return static_cast<double>(correct) / y.size();
    }
    
    std::shared_ptr<Optimizer> get_optimizer() const { return optimizer; }
    void set_optimizer(std::shared_ptr<Optimizer> new_optimizer) { optimizer = new_optimizer; }
};

// ----- Main Function -----
int main() {
    try {
        std::cout << "===== Extended Machine Learning Library Demo =====" << std::endl;
        
        // Sample data: features = [IQ, Study Time], target = exam score
        Matrix X(10, 2);
        Vector y(10);
        X[0][0] = 105; X[0][1] = 7.5; y[0] = 85;
        X[1][0] = 120; X[1][1] = 9.0; y[1] = 94;
        X[2][0] = 95;  X[2][1] = 3.5; y[2] = 70;
        X[3][0] = 110; X[3][1] = 5.0; y[3] = 88;
        X[4][0] = 130; X[4][1] = 8.0; y[4] = 96;
        X[5][0] = 115; X[5][1] = 6.5; y[5] = 87;
        X[6][0] = 98;  X[6][1] = 4.0; y[6] = 72;
        X[7][0] = 125; X[7][1] = 7.0; y[7] = 91;
        X[8][0] = 100; X[8][1] = 3.0; y[8] = 68;
        X[9][0] = 118; X[9][1] = 8.5; y[9] = 89;
        
        std::cout << "\n===== Linear Regression with Different Optimizers =====" << std::endl;
        
        // Train using SGD with Momentum
        std::cout << "\nSGD with Momentum:" << std::endl;
        auto sgd_optimizer = std::make_shared<SGDMomentum>(0.01, 0.9);
        LinearRegression sgd_model(0.01, 1000, 1e-6, true, true, sgd_optimizer);
        sgd_model.fit(X, y);
        Vector y_pred_sgd = sgd_model.predict(X);
        double r2_sgd = sgd_model.r_squared(X, y);
        std::cout << "R-squared: " << r2_sgd << std::endl;
        
        // Train using Adam
        std::cout << "\nAdam:" << std::endl;
        auto adam_optimizer = std::make_shared<Adam>(0.01);
        LinearRegression adam_model(0.01, 1000, 1e-6, true, true, adam_optimizer);
        adam_model.fit(X, y);
        Vector y_pred_adam = adam_model.predict(X);
        double r2_adam = adam_model.r_squared(X, y);
        std::cout << "R-squared: " << r2_adam << std::endl;
        
        // Train using RMSProp
        std::cout << "\nRMSProp:" << std::endl;
        auto rmsprop_optimizer = std::make_shared<RMSProp>(0.01);
        LinearRegression rmsprop_model(0.01, 1000, 1e-6, true, true, rmsprop_optimizer);
        rmsprop_model.fit(X, y);
        Vector y_pred_rmsprop = rmsprop_model.predict(X);
        double r2_rmsprop = rmsprop_model.r_squared(X, y);
        std::cout << "R-squared: " << r2_rmsprop << std::endl;
        
        // ----- Inference with Sample Input for Each Model -----
        // New sample: IQ = 115, Study Time = 8
        Matrix sample(1, 2);
        sample[0][0] = 115;
        sample[0][1] = 8.0;
        std::cout << "\n===== Inference on Sample Input (IQ = 115, Study Time = 8) =====" << std::endl;
        Vector sample_pred_sgd = sgd_model.predict(sample);
        std::cout << "SGD Model Prediction: " << sample_pred_sgd[0] << std::endl;
        Vector sample_pred_adam = adam_model.predict(sample);
        std::cout << "Adam Model Prediction: " << sample_pred_adam[0] << std::endl;
        Vector sample_pred_rmsprop = rmsprop_model.predict(sample);
        std::cout << "RMSProp Model Prediction: " << sample_pred_rmsprop[0] << std::endl;
        
        std::cout << "\n===== Cross-Validation =====" << std::endl;
        auto model_factory = []() {
            auto optimizer = std::make_shared<Adam>(0.01);
            return LinearRegression(0.01, 500, 1e-6, false, true, optimizer);
        };
        auto score_func = [](const LinearRegression& model, const Matrix& X, const Vector& y) {
            return model.r_squared(X, y);
        };
        auto cv_scores = cross_validate<LinearRegression>(X, y, 5, model_factory, score_func);
        std::cout << "5-fold cross-validation R-squared scores:" << std::endl;
        double mean_score = 0.0;
        for (size_t i = 0; i < cv_scores.size(); ++i) {
            std::cout << "  Fold " << i + 1 << ": " << cv_scores[i] << std::endl;
            mean_score += cv_scores[i];
        }
        mean_score /= cv_scores.size();
        std::cout << "Mean R-squared: " << mean_score << std::endl;
        
        std::cout << "\n===== Grid Search =====" << std::endl;
        struct HyperParams {
            double learning_rate;
            std::string optimizer_type;
            HyperParams() : learning_rate(0.01), optimizer_type("sgd") {}
            HyperParams(double lr, const std::string& opt) : learning_rate(lr), optimizer_type(opt) {}
        };
        std::vector<HyperParams> param_grid = {
            HyperParams(0.001, "sgd"),
            HyperParams(0.01, "sgd"),
            HyperParams(0.001, "adam"),
            HyperParams(0.01, "adam"),
            HyperParams(0.001, "rmsprop"),
            HyperParams(0.01, "rmsprop")
        };
        auto params_model_factory = [](const HyperParams& params) {
            std::shared_ptr<Optimizer> opt;
            if (params.optimizer_type == "sgd")
                opt = std::make_shared<SGDMomentum>(params.learning_rate);
            else if (params.optimizer_type == "adam")
                opt = std::make_shared<Adam>(params.learning_rate);
            else
                opt = std::make_shared<RMSProp>(params.learning_rate);
            return LinearRegression(params.learning_rate, 500, 1e-6, false, true, opt);
        };
        auto best_result = grid_search_cv<LinearRegression, HyperParams>(X, y, param_grid, params_model_factory, score_func, 5);
        std::cout << "Best parameters:" << std::endl;
        std::cout << "  Learning rate: " << best_result.first.learning_rate << std::endl;
        std::cout << "  Optimizer: " << best_result.first.optimizer_type << std::endl;
        std::cout << "Best R-squared: " << best_result.second << std::endl;
        
        // Create final best model
        std::shared_ptr<Optimizer> best_optimizer;
        if (best_result.first.optimizer_type == "sgd")
            best_optimizer = std::make_shared<SGDMomentum>(best_result.first.learning_rate);
        else if (best_result.first.optimizer_type == "adam")
            best_optimizer = std::make_shared<Adam>(best_result.first.learning_rate);
        else
            best_optimizer = std::make_shared<RMSProp>(best_result.first.learning_rate);
        
        LinearRegression best_model(best_result.first.learning_rate, 500, 1e-6, true, true, best_optimizer);
        std::cout << "\n===== Final Model Training with Best Hyperparameters =====" << std::endl;
        best_model.fit(X, y);
        Vector y_pred_best = best_model.predict(X);
        double final_r2 = best_model.r_squared(X, y);
        std::cout << "Final Model R-squared: " << final_r2 << std::endl;
        
        // Inference with the final best model
        Vector sample_pred_best = best_model.predict(sample);
        std::cout << "\nFinal Best Model Prediction on Sample Input (IQ = 115, Study Time = 8): " << sample_pred_best[0] << std::endl;
        
        std::cout << "\nDemo Completed Successfully." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
