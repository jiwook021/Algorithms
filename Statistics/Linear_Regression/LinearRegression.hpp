/**
 * @file LinearRegression.hpp
 * @brief Simple Linear Regression using the Ordinary Least Squares method.
 *
 * @details
 * Fits a line y = mx + b by minimizing the sum of squared residuals:
 *   m = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - (sum(x))^2)
 *   b = (sum(y) - m * sum(x)) / n
 */

#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <vector>

/**
 * @brief A data point with one feature and one output.
 */
struct DataPoint {
    double X;  ///< Feature
    double Y;  ///< Output
};

/**
 * @class LinearRegression
 * @brief Simple linear regression (y = mx + b).
 */
class LinearRegression {
public:
    LinearRegression();

    /**
     * @brief Fit the model using the least squares method.
     * @param dataset Training data points.
     */
    void Fit(const std::vector<DataPoint>& dataset);

    /**
     * @brief Predict y for a given x.
     * @param x Input value.
     * @return Predicted y.
     */
    double Predict(double x) const;

    /** @brief Get the learned slope. */
    double GetSlope() const;

    /** @brief Get the learned intercept. */
    double GetIntercept() const;

private:
    double m_;  ///< Slope
    double b_;  ///< Intercept
};

#endif // LINEAR_REGRESSION_HPP
