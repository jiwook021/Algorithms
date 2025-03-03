/**
 * @file main.cpp
 * @brief Driver for MLPerformanceThread
 */

#include "MLPerformanceThread.hpp"

int main() {
    // Demonstrate ThreadPool and thread-safe Vector/Matrix operations
    ThreadPool Pool(4);

    // Create a simple dataset
    const size_t NSamples = 100;
    const size_t NFeatures = 3;
    Matrix X(NSamples, NFeatures);
    Vector y(NSamples);

    std::mt19937 Rng(42);
    std::uniform_real_distribution<double> Dist(0.0, 1.0);

    for (size_t i = 0; i < NSamples; ++i) {
        for (size_t j = 0; j < NFeatures; ++j) {
            X[i][j] = Dist(Rng);
        }
        y[i] = X[i][0] * 2.0 + X[i][1] * 3.0 + X[i][2] * 1.5 + 0.1 * Dist(Rng);
    }

    // Train/test split
    auto [X_train, X_test, YTrain, YTest] = TrainTestSplit(X, y, 0.2);

    // Train linear regression with scaling
    LinearRegression Model(0.01, 1000, 1e-6, true, true, 4);
    Model.Fit(X_train, YTrain);

    // Evaluate
    Vector YPred = Model.Predict(X_test);
    double Mse = MeanSquaredError(YTest, YPred, Pool);
    double R2 = Model.RSquared(X_test, YTest);

    std::cout << "MSE: " << Mse << std::endl;
    std::cout << "R2:  " << R2 << std::endl;

    return 0;
}