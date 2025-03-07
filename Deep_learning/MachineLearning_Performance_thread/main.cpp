#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>

// Thread pool class for managing worker threads
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::vector<std::packaged_task<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

public:
    // Constructor creates the thread pool with specified number of threads
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : stop(false) {
        // Create worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::packaged_task<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        
                        // Wait until there's a task or the pool is stopped
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        // Exit if the pool is stopped and the task queue is empty
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        // Get the task from the front of the queue
                        task = std::move(this->tasks.back());
                        this->tasks.pop_back();
                    }
                    
                    // Execute the task
                    task();
                }
            });
        }
    }

    // Add a new task to the thread pool
    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        // Create a packaged task with the given function
        std::packaged_task<decltype(f())()> task(std::forward<F>(f));
        std::future<decltype(f())> result = task.get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Don't allow enqueueing after stopping the pool
            if (stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            
            // Wrap the packaged task into a void function
            std::packaged_task<void()> wrapper_task([task = std::move(task)]() mutable {
                task();
            });
            
            tasks.emplace_back(std::move(wrapper_task));
        }
        
        // Notify one waiting thread
        condition.notify_one();
        
        return result;
    }

    // Destructor cleans up and joins all threads
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        // Wake up all threads
        condition.notify_all();
        
        // Join all threads
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    // Get the number of threads in the pool
    size_t size() const {
        return workers.size();
    }
};

// Simple Vector class with thread-safe operations
class Vector {
private:
    std::vector<double> data;
    mutable std::mutex mtx;  // Mutex for thread-safe operations

public:
    // Default constructor
    Vector() : data() {}
    
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& vec) : data(vec) {}

    // Thread-safe element access
    double get(size_t index) const {
        std::lock_guard<std::mutex> lock(mtx);
        return data[index];
    }
    
    void set(size_t index, double value) {
        std::lock_guard<std::mutex> lock(mtx);
        data[index] = value;
    }
    
    // Non-thread safe element access, use with caution
    double& operator[](size_t index) { return data[index]; }
    const double& operator[](size_t index) const { return data[index]; }
    
    size_t size() const { 
        std::lock_guard<std::mutex> lock(mtx);
        return data.size(); 
    }

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

    // Thread-safe dot product operation
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

    // Thread-safe mean calculation
    double mean() const {
        std::lock_guard<std::mutex> lock(mtx);
        if (data.size() == 0) return 0.0;
        double sum = 0.0;
        for (const auto& val : data) {
            sum += val;
        }
        return sum / data.size();
    }

    // Thread-safe variance calculation
    double variance() const {
        std::lock_guard<std::mutex> lock(mtx);
        if (data.size() <= 1) return 0.0;
        double m = 0.0;
        for (const auto& val : data) {
            m += val;
        }
        m /= data.size();
        
        double sum_sq_diff = 0.0;
        for (const auto& val : data) {
            double diff = val - m;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / data.size();
    }

    double std_dev() const {
        return std::sqrt(variance());
    }

    // Thread-safe Pearson correlation coefficient with another vector
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

    const std::vector<double>& get_data() const { 
        std::lock_guard<std::mutex> lock(mtx);
        return data; 
    }
    
    // Add thread-safe element-wise addition
    void add_element(size_t index, double value) {
        std::lock_guard<std::mutex> lock(mtx);
        data[index] += value;
    }
};

// Simple Matrix class with thread-safe operations
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
    mutable std::mutex mtx;  // Mutex for thread-safe operations

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

    size_t num_rows() const { 
        std::lock_guard<std::mutex> lock(mtx);
        return rows; 
    }
    
    size_t num_cols() const { 
        std::lock_guard<std::mutex> lock(mtx);
        return cols; 
    }

    // Thread-safe column access
    Vector get_col(size_t col) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (col >= cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        Vector result(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i] = data[i][col];
        }
        return result;
    }

    // Thread-safe row access
    Vector get_row(size_t row) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (row >= rows) {
            throw std::out_of_range("Row index out of range");
        }
        
        return Vector(data[row]);
    }
    
    // Thread-safe element access
    double get(size_t row, size_t col) const {
        std::lock_guard<std::mutex> lock(mtx);
        return data[row][col];
    }
    
    void set(size_t row, size_t col, double value) {
        std::lock_guard<std::mutex> lock(mtx);
        data[row][col] = value;
    }
};

// Feature scaling (min-max scaling) with multithreading support
std::pair<Matrix, std::vector<std::pair<double, double>>> scale_features(const Matrix& X, ThreadPool& pool) {
    size_t n_samples = X.num_rows();
    size_t n_features = X.num_cols();
    
    // Find min and max values for each feature
    std::vector<std::pair<double, double>> min_max(n_features);  // (min, max) pairs
    std::vector<std::future<void>> futures;
    
    // Process each feature in a separate thread
    for (size_t j = 0; j < n_features; ++j) {
        futures.push_back(pool.enqueue([&X, &min_max, j, n_samples]() {
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::lowest();
            
            for (size_t i = 0; i < n_samples; ++i) {
                min_val = std::min(min_val, X[i][j]);
                max_val = std::max(max_val, X[i][j]);
            }
            
            min_max[j] = {min_val, max_val};
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Scale features
    Matrix X_scaled(n_samples, n_features);
    futures.clear();
    
    // Process each row in a separate thread
    for (size_t i = 0; i < n_samples; ++i) {
        futures.push_back(pool.enqueue([&X, &X_scaled, &min_max, i, n_features]() {
            for (size_t j = 0; j < n_features; ++j) {
                double range = min_max[j].second - min_max[j].first;
                if (range > 0) {
                    X_scaled[i][j] = (X[i][j] - min_max[j].first) / range;
                } else {
                    X_scaled[i][j] = 0.5;  // Default value if min == max
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    
    return {X_scaled, min_max};
}

// Linear Regression with gradient descent and multithreading support
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
    
    // Thread pool for parallel operations
    std::shared_ptr<ThreadPool> pool;
    
    // Mutex for thread-safe updates during gradient descent
    std::mutex weights_mutex;

public:
    LinearRegression(double learning_rate = 0.001, int max_iterations = 1000000, 
                     double tol = 1e-6, bool verbose = false, bool use_scaling = true,
                     size_t num_threads = 0)
        : learning_rate(learning_rate), max_iterations(max_iterations), 
          tol(tol), verbose(verbose), bias(0.0), use_scaling(use_scaling),
          target_min(0.0), target_max(1.0) {
              
        // Create a thread pool with specified or default number of threads
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        pool = std::make_shared<ThreadPool>(num_threads);
        
        if (verbose) {
            std::cout << "Created thread pool with " << num_threads << " threads" << std::endl;
        }
    }

    void fit(const Matrix& X_orig, const Vector& y_orig) {
        if (X_orig.num_rows() != y_orig.size() || X_orig.num_rows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        // Feature scaling if enabled
        Matrix X;
        Vector y;
        
        if (use_scaling) {
            if (verbose) {
                std::cout << "Scaling features..." << std::endl;
            }
            
            // Scale features using the thread pool
            auto [X_scaled, min_max] = scale_features(X_orig, *pool);
            X = X_scaled;
            feature_min_max = min_max;
            
            // Scale target
            target_min = *std::min_element(y_orig.get_data().begin(), y_orig.get_data().end());
            target_max = *std::max_element(y_orig.get_data().begin(), y_orig.get_data().end());
            
            double target_range = target_max - target_min;
            y = Vector(y_orig.size());
            
            // Process target scaling in parallel
            std::vector<std::future<void>> futures;
            size_t batch_size = std::max(size_t(1), y_orig.size() / pool->size());
            
            for (size_t i = 0; i < y_orig.size(); i += batch_size) {
                size_t end = std::min(i + batch_size, y_orig.size());
                futures.push_back(pool->enqueue([&, i, end, target_range]() {
                    for (size_t j = i; j < end; ++j) {
                        if (target_range > 0) {
                            y[j] = (y_orig[j] - target_min) / target_range;
                        } else {
                            y[j] = 0.5;
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.get();
            }
            
            if (verbose) {
                std::cout << "Feature scaling complete" << std::endl;
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

        if (verbose) {
            std::cout << "Starting gradient descent with " << n_samples << " samples and " 
                      << n_features << " features" << std::endl;
        }

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Compute predictions and loss in parallel
            Vector y_pred(n_samples);
            std::atomic<double> loss(0.0);
            std::vector<std::future<void>> futures;
            
            // Divide samples into batches for parallel processing
            size_t batch_size = std::max(size_t(1), n_samples / pool->size());
            
            for (size_t i = 0; i < n_samples; i += batch_size) {
                size_t end = std::min(i + batch_size, n_samples);
                futures.push_back(pool->enqueue([&, i, end]() {
                    double local_loss = 0.0;
                    
                    for (size_t j = i; j < end; ++j) {
                        // Compute prediction for this sample
                        double pred = bias;
                        for (size_t k = 0; k < n_features; ++k) {
                            pred += X[j][k] * weights[k];
                        }
                        y_pred[j] = pred;
                        
                        // Update local loss
                        double error = pred - y[j];
                        local_loss += error * error;
                    }
                    
                    // Atomic add to global loss
                    loss += local_loss;
                }));
            }
            
            // Wait for all batches to complete
            for (auto& future : futures) {
                future.get();
            }
            
            // Finalize loss computation
            loss = loss / n_samples;
            
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

            // Compute gradients in parallel
            Vector grad_w(n_features, 0.0);
            std::atomic<double> grad_b(0.0);
            futures.clear();
            
            for (size_t i = 0; i < n_samples; i += batch_size) {
                size_t end = std::min(i + batch_size, n_samples);
                futures.push_back(pool->enqueue([&, i, end]() {
                    std::vector<double> local_grad_w(n_features, 0.0);
                    double local_grad_b = 0.0;
                    
                    for (size_t j = i; j < end; ++j) {
                        double error = y_pred[j] - y[j];
                        
                        for (size_t k = 0; k < n_features; ++k) {
                            local_grad_w[k] += error * X[j][k];
                        }
                        local_grad_b += error;
                    }
                    
                    // Update global gradients with a lock to prevent race conditions
                    for (size_t k = 0; k < n_features; ++k) {
                        grad_w.add_element(k, local_grad_w[k]);
                    }
                    grad_b += local_grad_b;
                }));
            }
            
            // Wait for all gradient computations to complete
            for (auto& future : futures) {
                future.get();
            }
            
            // Scale gradients by number of samples
            for (size_t j = 0; j < n_features; ++j) {
                grad_w[j] /= n_samples;
            }
            grad_b /= n_samples;
            
            // Update with momentum - needs to be sequential as it depends on previous state
            {
                std::lock_guard<std::mutex> lock(weights_mutex);
                for (size_t j = 0; j < n_features; ++j) {
                    momentum[j] = beta * momentum[j] + (1.0 - beta) * grad_w[j];
                    weights[j] -= curr_learning_rate * momentum[j];
                }
                bias_momentum = beta * bias_momentum + (1.0 - beta) * grad_b;
                bias -= curr_learning_rate * bias_momentum;
            }
            
            // Debug output
            if (verbose && (iter % 1000 == 0 || iter == max_iterations - 1)) {
                std::cout << "Iteration " << iter << ": loss = " << loss << std::endl;
            }
        }
        
        if (verbose) {
            std::cout << "Gradient descent complete with final loss = " << prev_loss << std::endl;
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
            
            // Process feature scaling in parallel
            std::vector<std::future<void>> futures;
            size_t batch_size = std::max(size_t(1), n_samples / pool->size());
            
            for (size_t i = 0; i < n_samples; i += batch_size) {
                size_t end = std::min(i + batch_size, n_samples);
                futures.push_back(pool->enqueue([&, i, end]() {
                    for (size_t j = i; j < end; ++j) {
                        for (size_t k = 0; k < n_features; ++k) {
                            double range = feature_min_max[k].second - feature_min_max[k].first;
                            if (range > 0) {
                                X[j][k] = (X_orig[j][k] - feature_min_max[k].first) / range;
                            } else {
                                X[j][k] = 0.5;
                            }
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.get();
            }
        } else {
            X = X_orig;
        }
        
        size_t n_samples = X.num_rows();
        Vector y_pred(n_samples);

        // Make predictions in parallel
        std::vector<std::future<void>> futures;
        size_t batch_size = std::max(size_t(1), n_samples / pool->size());
        
        for (size_t i = 0; i < n_samples; i += batch_size) {
            size_t end = std::min(i + batch_size, n_samples);
            futures.push_back(pool->enqueue([&, i, end]() {
                for (size_t j = i; j < end; ++j) {
                    double pred = bias;
                    for (size_t k = 0; k < X.num_cols(); ++k) {
                        pred += X[j][k] * weights[k];
                    }
                    y_pred[j] = pred;
                }
            }));
        }
        
        // Wait for all prediction computations to complete
        for (auto& future : futures) {
            future.get();
        }
        
        // Unscale predictions if needed
        if (use_scaling) {
            double target_range = target_max - target_min;
            
            // Process prediction unscaling in parallel
            futures.clear();
            
            for (size_t i = 0; i < n_samples; i += batch_size) {
                size_t end = std::min(i + batch_size, n_samples);
                futures.push_back(pool->enqueue([&, i, end, target_range]() {
                    for (size_t j = i; j < end; ++j) {
                        if (target_range > 0) {
                            y_pred[j] = y_pred[j] * target_range + target_min;
                        } else {
                            y_pred[j] = target_min;
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.get();
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

    // Calculate R-squared with multithreading
    double r_squared(const Matrix& X, const Vector& y) const {
        if (X.num_rows() != y.size() || X.num_rows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        Vector y_pred = predict(X);
        double y_mean = y.mean();
        
        std::atomic<double> ss_total(0.0);
        std::atomic<double> ss_residual(0.0);
        
        // Process R-squared calculation in parallel
        std::vector<std::future<void>> futures;
        size_t n_samples = X.num_rows();
        size_t batch_size = std::max(size_t(1), n_samples / pool->size());
        
        for (size_t i = 0; i < n_samples; i += batch_size) {
            size_t end = std::min(i + batch_size, n_samples);
            futures.push_back(pool->enqueue([&, i, end]() {
                double local_ss_total = 0.0;
                double local_ss_residual = 0.0;
                
                for (size_t j = i; j < end; ++j) {
                    double diff_total = y[j] - y_mean;
                    double diff_residual = y[j] - y_pred[j];
                    
                    local_ss_total += diff_total * diff_total;
                    local_ss_residual += diff_residual * diff_residual;
                }
                
                // Atomic updates to global sums
                ss_total += local_ss_total;
                ss_residual += local_ss_residual;
            }));
        }
        
        // Wait for all computations to complete
        for (auto& future : futures) {
            future.get();
        }
        
        if (ss_total == 0.0) {
            return 0.0;  // Avoid division by zero
        }
        
        return 1.0 - (ss_residual / ss_total);
    }
};

// Calculate Mean Squared Error with multithreading
double mean_squared_error(const Vector& y_true, const Vector& y_pred, ThreadPool& pool) {
    if (y_true.size() != y_pred.size() || y_true.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    std::atomic<double> sum(0.0);
    std::vector<std::future<void>> futures;
    size_t n_samples = y_true.size();
    size_t batch_size = std::max(size_t(1), n_samples / pool.size());
    
    for (size_t i = 0; i < n_samples; i += batch_size) {
        size_t end = std::min(i + batch_size, n_samples);
        futures.push_back(pool.enqueue([&, i, end]() {
            double local_sum = 0.0;
            for (size_t j = i; j < end; ++j) {
                double diff = y_true[j] - y_pred[j];
                local_sum += diff * diff;
            }
            sum += local_sum;
        }));
    }
    
    for (auto& future : futures) {
        future.get();
    }
    
    return sum / n_samples;
}
};

// Multiple Linear Regression model (identical interface to LinearRegression)
typedef LinearRegression MultipleLinearRegression;

// Standardize features (z-score normalization) with multithreading
Matrix standardize(const Matrix& X, ThreadPool& pool) {
    size_t n_samples = X.num_rows();
    size_t n_features = X.num_cols();
    
    Matrix X_std(n_samples, n_features);
    std::vector<std::future<void>> futures;
    
    // Process each feature in a separate thread
    for (size_t j = 0; j < n_features; ++j) {
        futures.push_back(pool.enqueue([&X, &X_std, j, n_samples]() {
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
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    
    return X_std;
}

// Split data into training and testing sets - this is already efficiently implemented
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