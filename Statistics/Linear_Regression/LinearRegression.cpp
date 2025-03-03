/**
 * @file LinearRegression.cpp
 * @brief Implementation of simple linear regression.
 */

#include "LinearRegression.hpp"

LinearRegression::LinearRegression() : m_(0.0), b_(0.0) {}

void LinearRegression::Fit(const std::vector<DataPoint>& dataset) {
    double xSum = 0.0, ySum = 0.0, xySum = 0.0, x2Sum = 0.0;
    double n = static_cast<double>(dataset.size());

    for (const auto& dp : dataset) {
        xSum += dp.X;
        ySum += dp.Y;
        xySum += dp.X * dp.Y;
        x2Sum += dp.X * dp.X;
    }

    m_ = (n * xySum - xSum * ySum) / (n * x2Sum - xSum * xSum);
    b_ = (ySum - m_ * xSum) / n;
}

double LinearRegression::Predict(double x) const {
    return m_ * x + b_;
}

double LinearRegression::GetSlope() const { return m_; }
double LinearRegression::GetIntercept() const { return b_; }
