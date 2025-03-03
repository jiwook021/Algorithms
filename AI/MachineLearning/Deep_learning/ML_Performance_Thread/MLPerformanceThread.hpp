/**
 * @file MLPerformanceThread.hpp
 * @brief Multithreaded ML training performance benchmark
 * @details Benchmarks neural network training throughput using std::thread parallelism. Measures epoch time, throughput, and convergence across varying thread counts and batch sizes.
 */

#pragma once

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
    std::vector<std::thread> Workers;
    std::vector<std::packaged_task<void()>> Tasks;
    std::mutex QueueMutex;
    std::condition_variable Condition;
    std::atomic<bool> Stop;

public:
    // Constructor creates the thread pool with specified number of threads
    ThreadPool(size_t NumThreads = std::thread::hardware_concurrency()) : Stop(false) {
        // Create worker threads
        for (size_t i = 0; i < NumThreads; ++i) {
            Workers.emplace_back([this] {
                while (true) {
                    std::packaged_task<void()> Task;
                    
                    {
                        std::unique_lock<std::mutex> Lock(QueueMutex);
                        
                        // Wait until there's a task or the pool is stopped
                        this->Condition.wait(lock, [this] { 
                            return this->Stop || !this->Tasks.empty(); 
                        });
                        
                        // Exit if the pool is stopped and the task queue is empty
                        if (this->Stop && this->Tasks.empty()) {
                            return;
                        }
                        
                        // Get the task from the front of the queue
                        Task = std::move(this->Tasks.back());
                        this->Tasks.pop_back();
                    }
                    
                    // Execute the task
                    Task();
                }
            });
        }
    }

    // Add a new task to the thread pool
    template<class F>
    auto Enqueue(F&& f) -> std::future<decltype(f())> {
        // Create a packaged task with the given function
        std::packaged_task<decltype(f())()> Task(std::forward<F>(f));
        std::future<decltype(f())> Result = Task.get_future();
        
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            
            // Don't allow enqueueing after stopping the pool
            if (Stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            
            // Wrap the packaged task into a void function
            std::packaged_task<void()> WrapperTask([Task = std::move(Task)]() mutable {
                Task();
            });
            
            Tasks.emplace_back(std::move(WrapperTask));
        }
        
        // Notify one waiting thread
        Condition.notify_one();
        
        return Result;
    }

    // Destructor cleans up and joins all threads
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            Stop = true;
        }
        
        // Wake up all threads
        Condition.notify_all();
        
        // Join all threads
        for (std::thread &Worker : Workers) {
            if (Worker.joinable()) {
                Worker.join();
            }
        }
    }
    
    // Get the number of threads in the pool
    size_t size() const {
        return Workers.size();
    }
};

// Simple Vector class with thread-safe operations
class Vector {
private:
    std::vector<double> data;
    mutable std::mutex Mtx;  // Mutex for thread-safe operations

public:
    // Default constructor
    Vector() : data() {}
    
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& Vec) : data(Vec) {}

    // Copy constructor
    Vector(const Vector& Other) {
        std::lock_guard<std::mutex> Lock(Other.Mtx);
        data = Other.data;
    }

    // Move constructor
    Vector(Vector&& Other) noexcept {
        std::lock_guard<std::mutex> Lock(Other.Mtx);
        data = std::move(Other.data);
    }

    // Copy assignment
    Vector& operator=(const Vector& Other) {
        if (this != &Other) {
            std::scoped_lock lock(Mtx, Other.Mtx);
            data = Other.data;
        }
        return *this;
    }

    // Move assignment
    Vector& operator=(Vector&& Other) noexcept {
        if (this != &Other) {
            std::scoped_lock lock(Mtx, Other.Mtx);
            data = std::move(Other.data);
        }
        return *this;
    }

    // Thread-safe element access
    double get(size_t Index) const {
        std::lock_guard<std::mutex> Lock(Mtx);
        return data[Index];
    }
    
    void set(size_t Index, double value) {
        std::lock_guard<std::mutex> Lock(Mtx);
        data[Index] = value;
    }
    
    // Non-thread safe element access, use with caution
    double& operator[](size_t Index) { return data[Index]; }
    const double& operator[](size_t Index) const { return data[Index]; }
    
    size_t size() const { 
        std::lock_guard<std::mutex> Lock(Mtx);
        return data.size(); 
    }

    Vector operator+(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for addition");
        }
        
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] + Other[i];
        }
        return Result;
    }

    Vector operator-(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for subtraction");
        }
        
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] - Other[i];
        }
        return Result;
    }

    Vector operator*(double Scalar) const {
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] * Scalar;
        }
        return Result;
    }

    // Thread-safe dot product operation
    double Dot(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        
        double Result = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            Result += data[i] * Other[i];
        }
        return Result;
    }

    // Thread-safe mean calculation
    double Mean() const {
        std::lock_guard<std::mutex> Lock(Mtx);
        if (data.size() == 0) return 0.0;
        double Sum = 0.0;
        for (const auto& Val : data) {
            Sum += Val;
        }
        return Sum / data.size();
    }

    // Thread-safe variance calculation
    double Variance() const {
        std::lock_guard<std::mutex> Lock(Mtx);
        if (data.size() <= 1) return 0.0;
        double m = 0.0;
        for (const auto& Val : data) {
            m += Val;
        }
        m /= data.size();
        
        double SumSqDiff = 0.0;
        for (const auto& Val : data) {
            double Diff = Val - m;
            SumSqDiff += Diff * Diff;
        }
        return SumSqDiff / data.size();
    }

    double StdDev() const {
        return std::sqrt(Variance());
    }

    // Thread-safe Pearson correlation coefficient with another vector
    double Correlation(const Vector& Other) const {
        if (size() != Other.size() || size() == 0) {
            throw std::invalid_argument("Vectors must have the same non-zero size");
        }

        double MeanX = Mean();
        double MeanY = Other.Mean();
        double SumXy = 0.0, SumX2 = 0.0, SumY2 = 0.0;

        for (size_t i = 0; i < size(); ++i) {
            double XDiff = data[i] - MeanX;
            double YDiff = Other[i] - MeanY;
            SumXy += XDiff * YDiff;
            SumX2 += XDiff * XDiff;
            SumY2 += YDiff * YDiff;
        }

        if (SumX2 == 0.0 || SumY2 == 0.0) {
            return 0.0;  // Avoid division by zero
        }

        return SumXy / std::sqrt(SumX2 * SumY2);
    }

    const std::vector<double>& GetData() const { 
        std::lock_guard<std::mutex> Lock(Mtx);
        return data; 
    }
    
    // Add thread-safe element-wise addition
    void AddElement(size_t Index, double value) {
        std::lock_guard<std::mutex> Lock(Mtx);
        data[Index] += value;
    }
};

// Simple Matrix class with thread-safe operations
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t Rows;
    size_t Cols;
    mutable std::mutex Mtx;  // Mutex for thread-safe operations

public:
    // Default constructor
    Matrix() : data(), Rows(0), Cols(0) {}
    
    Matrix(size_t Rows, size_t Cols, double value = 0.0)
        : data(Rows, std::vector<double>(Cols, value)), Rows(Rows), Cols(Cols) {}

    Matrix(const std::vector<std::vector<double>>& mat) {
        Rows = mat.size();
        Cols = Rows > 0 ? mat[0].size() : 0;
        data = mat;
    }

    // Copy constructor
    Matrix(const Matrix& Other) {
        std::lock_guard<std::mutex> Lock(Other.Mtx);
        data = Other.data;
        Rows = Other.Rows;
        Cols = Other.Cols;
    }
    // Move constructor
    Matrix(Matrix&& Other) noexcept {
        std::lock_guard<std::mutex> Lock(Other.Mtx);
        data = std::move(Other.data);
        Rows = Other.Rows;
        Cols = Other.Cols;
        Other.Rows = 0;
        Other.Cols = 0;
    }
    // Copy assignment
    Matrix& operator=(const Matrix& Other) {
        if (this != &Other) {
            std::scoped_lock lock(Mtx, Other.Mtx);
            data = Other.data;
            Rows = Other.Rows;
            Cols = Other.Cols;
        }
        return *this;
    }
    // Move assignment
    Matrix& operator=(Matrix&& Other) noexcept {
        if (this != &Other) {
            std::scoped_lock lock(Mtx, Other.Mtx);
            data = std::move(Other.data);
            Rows = Other.Rows;
            Cols = Other.Cols;
            Other.Rows = 0;
            Other.Cols = 0;
        }
        return *this;
    }

    std::vector<double>& operator[](size_t Row) { return data[Row]; }
    const std::vector<double>& operator[](size_t Row) const { return data[Row]; }

    size_t NumRows() const { 
        std::lock_guard<std::mutex> Lock(Mtx);
        return Rows; 
    }
    
    size_t NumCols() const { 
        std::lock_guard<std::mutex> Lock(Mtx);
        return Cols; 
    }

    // Thread-safe column access
    Vector GetCol(size_t Col) const {
        std::lock_guard<std::mutex> Lock(Mtx);
        if (Col >= Cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        Vector Result(Rows);
        for (size_t i = 0; i < Rows; ++i) {
            Result[i] = data[i][Col];
        }
        return Result;
    }

    // Thread-safe row access
    Vector GetRow(size_t Row) const {
        std::lock_guard<std::mutex> Lock(Mtx);
        if (Row >= Rows) {
            throw std::out_of_range("Row index out of range");
        }
        
        return Vector(data[Row]);
    }
    
    // Thread-safe element access
    double get(size_t Row, size_t Col) const {
        std::lock_guard<std::mutex> Lock(Mtx);
        return data[Row][Col];
    }
    
    void set(size_t Row, size_t Col, double value) {
        std::lock_guard<std::mutex> Lock(Mtx);
        data[Row][Col] = value;
    }
};

// Feature scaling (min-max scaling) with multithreading support
std::pair<Matrix, std::vector<std::pair<double, double>>> ScaleFeatures(const Matrix& X, ThreadPool& Pool) {
    size_t NSamples = X.NumRows();
    size_t NFeatures = X.NumCols();
    
    // Find min and max values for each feature
    std::vector<std::pair<double, double>> MinMax(NFeatures);  // (min, max) pairs
    std::vector<std::future<void>> Futures;
    
    // Process each feature in a separate thread
    for (size_t j = 0; j < NFeatures; ++j) {
        Futures.push_back(Pool.Enqueue([&X, &MinMax, j, NSamples]() {
            double MinVal = std::numeric_limits<double>::max();
            double MaxVal = std::numeric_limits<double>::lowest();
            
            for (size_t i = 0; i < NSamples; ++i) {
                MinVal = std::min(MinVal, X[i][j]);
                MaxVal = std::max(MaxVal, X[i][j]);
            }
            
            MinMax[j] = {MinVal, MaxVal};
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : Futures) {
        future.get();
    }
    
    // Scale features
    Matrix X_scaled(NSamples, NFeatures);
    Futures.clear();
    
    // Process each row in a separate thread
    for (size_t i = 0; i < NSamples; ++i) {
        Futures.push_back(Pool.Enqueue([&X, &X_scaled, &MinMax, i, NFeatures]() {
            for (size_t j = 0; j < NFeatures; ++j) {
                double Range = MinMax[j].second - MinMax[j].first;
                if (Range > 0) {
                    X_scaled[i][j] = (X[i][j] - MinMax[j].first) / Range;
                } else {
                    X_scaled[i][j] = 0.5;  // Default value if min == max
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : Futures) {
        future.get();
    }
    
    return {X_scaled, MinMax};
}

// Linear Regression with gradient descent and multithreading support
class LinearRegression {
private:
    Vector Weights;
    double Bias;
    double LearningRate;
    int MaxIterations;
    double Tol;
    bool Verbose;
    
    // For feature scaling
    std::vector<std::pair<double, double>> FeatureMinMax;
    double TargetMin;
    double TargetMax;
    bool UseScaling;
    
    // Thread pool for parallel operations
    std::shared_ptr<ThreadPool> Pool;
    
    // Mutex for thread-safe updates during gradient descent
    std::mutex WeightsMutex;

public:
    LinearRegression(double LearningRate = 0.001, int MaxIterations = 1000000, 
                     double Tol = 1e-6, bool Verbose = false, bool UseScaling = true,
                     size_t NumThreads = 0)
        : LearningRate(LearningRate), MaxIterations(MaxIterations), 
          Tol(Tol), Verbose(Verbose), Bias(0.0), UseScaling(UseScaling),
          TargetMin(0.0), TargetMax(1.0) {
              
        // Create a thread pool with specified or default number of threads
        if (NumThreads == 0) {
            NumThreads = std::thread::hardware_concurrency();
        }
        Pool = std::make_shared<ThreadPool>(NumThreads);
        
        if (Verbose) {
            std::cout << "Created thread pool with " << NumThreads << " threads" << std::endl;
        }
    }

    void Fit(const Matrix& X_orig, const Vector& YOrig) {
        if (X_orig.NumRows() != YOrig.size() || X_orig.NumRows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        // Feature scaling if enabled
        Matrix X;
        Vector y;
        
        if (UseScaling) {
            if (Verbose) {
                std::cout << "Scaling features..." << std::endl;
            }
            
            // Scale features using the thread pool
            auto [X_scaled, MinMax] = ScaleFeatures(X_orig, *Pool);
            X = X_scaled;
            FeatureMinMax = MinMax;
            
            // Scale target
            TargetMin = *std::min_element(YOrig.GetData().begin(), YOrig.GetData().end());
            TargetMax = *std::max_element(YOrig.GetData().begin(), YOrig.GetData().end());
            
            double TargetRange = TargetMax - TargetMin;
            y = Vector(YOrig.size());
            
            // Process target scaling in parallel
            std::vector<std::future<void>> Futures;
            size_t BatchSize = std::max(size_t(1), YOrig.size() / Pool->size());
            
            for (size_t i = 0; i < YOrig.size(); i += BatchSize) {
                size_t end = std::min(i + BatchSize, YOrig.size());
                Futures.push_back(Pool->Enqueue([&, i, end, TargetRange]() {
                    for (size_t j = i; j < end; ++j) {
                        if (TargetRange > 0) {
                            y[j] = (YOrig[j] - TargetMin) / TargetRange;
                        } else {
                            y[j] = 0.5;
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : Futures) {
                future.get();
            }
            
            if (Verbose) {
                std::cout << "Feature scaling complete" << std::endl;
            }
        } else {
            X = X_orig;
            y = YOrig;
        }
        
        size_t NSamples = X.NumRows();
        size_t NFeatures = X.NumCols();

        // Initialize weights and bias
        Weights = Vector(NFeatures, 0.0);
        Bias = 0.0;

        double PrevLoss = std::numeric_limits<double>::max();
        double CurrLearningRate = LearningRate;
        
        // Adaptive momentum
        Vector Momentum(NFeatures, 0.0);
        double BiasMomentum = 0.0;
        double Beta = 0.9;  // Momentum factor

        if (Verbose) {
            std::cout << "Starting gradient descent with " << NSamples << " samples and " 
                      << NFeatures << " features" << std::endl;
        }

        for (int Iter = 0; Iter < MaxIterations; ++Iter) {
            // Compute predictions and loss in parallel
            Vector YPred(NSamples);
            std::atomic<double> Loss(0.0);
            std::vector<std::future<void>> Futures;
            
            // Divide samples into batches for parallel processing
            size_t BatchSize = std::max(size_t(1), NSamples / Pool->size());
            
            for (size_t i = 0; i < NSamples; i += BatchSize) {
                size_t end = std::min(i + BatchSize, NSamples);
                Futures.push_back(Pool->Enqueue([&, i, end]() {
                    double LocalLoss = 0.0;
                    
                    for (size_t j = i; j < end; ++j) {
                        // Compute prediction for this sample
                        double Pred = Bias;
                        for (size_t k = 0; k < NFeatures; ++k) {
                            Pred += X[j][k] * Weights[k];
                        }
                        YPred[j] = Pred;
                        
                        // Update local loss
                        double Error = Pred - y[j];
                        LocalLoss += Error * Error;
                    }
                    
                    // Atomic add to global loss
                    double Expected = Loss.load();
                    while (!Loss.compare_exchange_weak(Expected, Expected + LocalLoss));
                }));
            }
            
            // Wait for all batches to complete
            for (auto& future : Futures) {
                future.get();
            }
            
            // Finalize loss computation
            Loss.store(Loss.load() / static_cast<double>(NSamples));
            double LossVal = Loss.load();
            
            // Check for convergence
            if (std::abs(LossVal - PrevLoss) < Tol) {
                if (Verbose) {
                    std::cout << "Converged at iteration " << Iter << " with loss " << LossVal << std::endl;
                }
                break;
            }
            
            // Learning rate scheduling
            // Reduce learning rate if loss increases
            if (LossVal > PrevLoss) {
                CurrLearningRate *= 0.5;
                if (Verbose) {
                    std::cout << "Reducing learning rate to " << CurrLearningRate << std::endl;
                }
            }
            
            PrevLoss = LossVal;

            // Compute gradients in parallel
            Vector GradW(NFeatures, 0.0);
            std::atomic<double> GradB(0.0);
            Futures.clear();
            
            for (size_t i = 0; i < NSamples; i += BatchSize) {
                size_t end = std::min(i + BatchSize, NSamples);
                Futures.push_back(Pool->Enqueue([&, i, end]() {
                    std::vector<double> LocalGradW(NFeatures, 0.0);
                    double LocalGradB = 0.0;
                    
                    for (size_t j = i; j < end; ++j) {
                        double Error = YPred[j] - y[j];
                        
                        for (size_t k = 0; k < NFeatures; ++k) {
                            LocalGradW[k] += Error * X[j][k];
                        }
                        LocalGradB += Error;
                    }
                    
                    // Update global gradients with a lock to prevent race conditions
                    for (size_t k = 0; k < NFeatures; ++k) {
                        GradW.AddElement(k, LocalGradW[k]);
                    }
                    double ExpGradB = GradB.load();
                    while (!GradB.compare_exchange_weak(ExpGradB, ExpGradB + LocalGradB));
                }));
            }
            
            // Wait for all gradient computations to complete
            for (auto& future : Futures) {
                future.get();
            }
            
            // Scale gradients by number of samples
            for (size_t j = 0; j < NFeatures; ++j) {
                GradW[j] /= NSamples;
            }
            GradB.store(GradB.load() / static_cast<double>(NSamples));
            
            // Update with momentum - needs to be sequential as it depends on previous state
            {
                std::lock_guard<std::mutex> Lock(WeightsMutex);
                for (size_t j = 0; j < NFeatures; ++j) {
                    Momentum[j] = Beta * Momentum[j] + (1.0 - Beta) * GradW[j];
                    Weights[j] -= CurrLearningRate * Momentum[j];
                }
                BiasMomentum = Beta * BiasMomentum + (1.0 - Beta) * GradB.load();
                Bias -= CurrLearningRate * BiasMomentum;
            }
            
            // Debug output
            if (Verbose && (Iter % 1000 == 0 || Iter == MaxIterations - 1)) {
                std::cout << "Iteration " << Iter << ": loss = " << LossVal << std::endl;
            }
        }
        
        if (Verbose) {
            std::cout << "Gradient descent complete with final loss = " << PrevLoss << std::endl;
        }
    }

    Vector Predict(const Matrix& X_orig) const {
        if (X_orig.NumCols() != Weights.size()) {
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        }
        
        // Scale input features if needed
        Matrix X;
        if (UseScaling && !FeatureMinMax.empty()) {
            size_t NSamples = X_orig.NumRows();
            size_t NFeatures = X_orig.NumCols();
            
            X = Matrix(NSamples, NFeatures);
            
            // Process feature scaling in parallel
            std::vector<std::future<void>> Futures;
            size_t BatchSize = std::max(size_t(1), NSamples / Pool->size());
            
            for (size_t i = 0; i < NSamples; i += BatchSize) {
                size_t end = std::min(i + BatchSize, NSamples);
                Futures.push_back(Pool->Enqueue([&, i, end]() {
                    for (size_t j = i; j < end; ++j) {
                        for (size_t k = 0; k < NFeatures; ++k) {
                            double Range = FeatureMinMax[k].second - FeatureMinMax[k].first;
                            if (Range > 0) {
                                X[j][k] = (X_orig[j][k] - FeatureMinMax[k].first) / Range;
                            } else {
                                X[j][k] = 0.5;
                            }
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : Futures) {
                future.get();
            }
        } else {
            X = X_orig;
        }
        
        size_t NSamples = X.NumRows();
        Vector YPred(NSamples);

        // Make predictions in parallel
        std::vector<std::future<void>> Futures;
        size_t BatchSize = std::max(size_t(1), NSamples / Pool->size());
        
        for (size_t i = 0; i < NSamples; i += BatchSize) {
            size_t end = std::min(i + BatchSize, NSamples);
            Futures.push_back(Pool->Enqueue([&, i, end]() {
                for (size_t j = i; j < end; ++j) {
                    double Pred = Bias;
                    for (size_t k = 0; k < X.NumCols(); ++k) {
                        Pred += X[j][k] * Weights[k];
                    }
                    YPred[j] = Pred;
                }
            }));
        }
        
        // Wait for all prediction computations to complete
        for (auto& future : Futures) {
            future.get();
        }
        
        // Unscale predictions if needed
        if (UseScaling) {
            double TargetRange = TargetMax - TargetMin;
            
            // Process prediction unscaling in parallel
            Futures.clear();
            
            for (size_t i = 0; i < NSamples; i += BatchSize) {
                size_t end = std::min(i + BatchSize, NSamples);
                Futures.push_back(Pool->Enqueue([&, i, end, TargetRange]() {
                    for (size_t j = i; j < end; ++j) {
                        if (TargetRange > 0) {
                            YPred[j] = YPred[j] * TargetRange + TargetMin;
                        } else {
                            YPred[j] = TargetMin;
                        }
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : Futures) {
                future.get();
            }
        }

        return YPred;
    }

    Vector GetWeights() const {
        return Weights;
    }

    double GetBias() const {
        return Bias;
    }

    // Calculate R-squared with multithreading
    double RSquared(const Matrix& X, const Vector& y) const {
        if (X.NumRows() != y.size() || X.NumRows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        Vector YPred = Predict(X);
        double YMean = y.Mean();
        
        std::atomic<double> SsTotal(0.0);
        std::atomic<double> SsResidual(0.0);
        
        // Process R-squared calculation in parallel
        std::vector<std::future<void>> Futures;
        size_t NSamples = X.NumRows();
        size_t BatchSize = std::max(size_t(1), NSamples / Pool->size());
        
        for (size_t i = 0; i < NSamples; i += BatchSize) {
            size_t end = std::min(i + BatchSize, NSamples);
            Futures.push_back(Pool->Enqueue([&, i, end]() {
                double LocalSsTotal = 0.0;
                double LocalSsResidual = 0.0;
                
                for (size_t j = i; j < end; ++j) {
                    double DiffTotal = y[j] - YMean;
                    double DiffResidual = y[j] - YPred[j];
                    
                    LocalSsTotal += DiffTotal * DiffTotal;
                    LocalSsResidual += DiffResidual * DiffResidual;
                }
                
                // Atomic updates to global sums
                double ExpSsT = SsTotal.load();
                while (!SsTotal.compare_exchange_weak(ExpSsT, ExpSsT + LocalSsTotal));
                double ExpSsR = SsResidual.load();
                while (!SsResidual.compare_exchange_weak(ExpSsR, ExpSsR + LocalSsResidual));
            }));
        }
        
        // Wait for all computations to complete
        for (auto& future : Futures) {
            future.get();
        }
        
        if (SsTotal.load() == 0.0) {
            return 0.0;  // Avoid division by zero
        }
        
        return 1.0 - (SsResidual.load() / SsTotal.load());
    }
};

// Calculate Mean Squared Error with multithreading
double MeanSquaredError(const Vector& YTrue, const Vector& YPred, ThreadPool& Pool) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    std::atomic<double> Sum(0.0);
    std::vector<std::future<void>> Futures;
    size_t NSamples = YTrue.size();
    size_t BatchSize = std::max(size_t(1), NSamples / Pool.size());
    
    for (size_t i = 0; i < NSamples; i += BatchSize) {
        size_t end = std::min(i + BatchSize, NSamples);
        Futures.push_back(Pool.Enqueue([&, i, end]() {
            double LocalSum = 0.0;
            for (size_t j = i; j < end; ++j) {
                double Diff = YTrue[j] - YPred[j];
                LocalSum += Diff * Diff;
            }
            double ExpSum = Sum.load();
            while (!Sum.compare_exchange_weak(ExpSum, ExpSum + LocalSum));
        }));
    }
    
    for (auto& future : Futures) {
        future.get();
    }
    
    return Sum.load() / static_cast<double>(NSamples);
}

// Multiple Linear Regression model (identical interface to LinearRegression)
typedef LinearRegression MultipleLinearRegression;

// Standardize features (z-score normalization) with multithreading
Matrix Standardize(const Matrix& X, ThreadPool& Pool) {
    size_t NSamples = X.NumRows();
    size_t NFeatures = X.NumCols();
    
    Matrix X_std(NSamples, NFeatures);
    std::vector<std::future<void>> Futures;
    
    // Process each feature in a separate thread
    for (size_t j = 0; j < NFeatures; ++j) {
        Futures.push_back(Pool.Enqueue([&X, &X_std, j, NSamples]() {
            Vector Feature = X.GetCol(j);
            double Mean = Feature.Mean();
            double StdDev = Feature.StdDev();
            
            for (size_t i = 0; i < NSamples; ++i) {
                if (StdDev > 0) {
                    X_std[i][j] = (X[i][j] - Mean) / StdDev;
                } else {
                    X_std[i][j] = 0.0;
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : Futures) {
        future.get();
    }
    
    return X_std;
}

// Split data into training and testing sets - this is already efficiently implemented
std::tuple<Matrix, Matrix, Vector, Vector> TrainTestSplit(
    const Matrix& X, const Vector& y, double TestSize = 0.2) {
    
    if (X.NumRows() != y.size() || X.NumRows() == 0) {
        throw std::invalid_argument("Invalid input data dimensions");
    }
    
    size_t NSamples = X.NumRows();
    size_t NTest = static_cast<size_t>(NSamples * TestSize);
    
    // Ensure at least one test sample
    NTest = std::max(size_t(1), NTest);
    
    // Ensure at least one training sample
    size_t NTrain = NSamples - NTest;
    NTrain = std::max(size_t(1), NTrain);
    
    // Adjust n_test if necessary
    NTest = NSamples - NTrain;
    
    // Create indices and shuffle them
    std::vector<size_t> Indices(NSamples);
    for (size_t i = 0; i < NSamples; ++i) {
        Indices[i] = i;
    }
    
    std::random_device Rd;
    std::mt19937 g(Rd());
    std::shuffle(Indices.begin(), Indices.end(), g);
    
    // Split data
    Matrix X_train(NTrain, X.NumCols());
    Matrix X_test(NTest, X.NumCols());
    Vector YTrain(NTrain);
    Vector YTest(NTest);
    
    for (size_t i = 0; i < NTrain; ++i) {
        size_t Idx = Indices[i];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_train[i][j] = X[Idx][j];
        }
        YTrain[i] = y[Idx];
    }
    
    for (size_t i = 0; i < NTest; ++i) {
        size_t Idx = Indices[i + NTrain];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_test[i][j] = X[Idx][j];
        }
        YTest[i] = y[Idx];
    }
    
    return {X_train, X_test, YTrain, YTest};
}

