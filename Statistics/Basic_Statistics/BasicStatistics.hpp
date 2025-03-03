/**
 * @file BasicStatistics.hpp
 * @brief Basic statistical functions: mean, slope, intercept, prediction, MSE.
 *
 * @details
 * Provides fundamental statistical computations for simple linear regression
 * using the method of least squares.
 *
 * Formulas:
 *   - Mean:      mu = (1/n) * sum(x_i)
 *   - Slope:     m = sum((x_i - mu_x)(y_i - mu_y)) / sum((x_i - mu_x)^2)
 *   - Intercept: b = mu_y - m * mu_x
 *   - MSE:       (1/n) * sum((y_i - (m*x_i + b))^2)
 */

#ifndef BASIC_STATISTICS_HPP
#define BASIC_STATISTICS_HPP

#include <vector>

/**
 * @class BasicStatistics
 * @brief Encapsulates basic statistical computations for univariate data.
 */
class BasicStatistics {
public:
    /**
     * @brief Compute the arithmetic mean of a data vector.
     * @param data Input values.
     * @return Mean value.
     */
    static double Mean(const std::vector<double>& data);

    /**
     * @brief Compute the slope of the least-squares regression line.
     * @param x Independent variable values.
     * @param y Dependent variable values.
     * @param meanX Mean of x values.
     * @param meanY Mean of y values.
     * @return Slope (m).
     */
    static double Slope(const std::vector<double>& x,
                        const std::vector<double>& y,
                        double meanX, double meanY);

    /**
     * @brief Compute the y-intercept of the regression line.
     * @param meanY Mean of y values.
     * @param m Slope.
     * @param meanX Mean of x values.
     * @return Intercept (b).
     */
    static double Intercept(double meanY, double m, double meanX);

    /**
     * @brief Predict y for a given x using y = mx + b.
     * @param x Input value.
     * @param m Slope.
     * @param b Intercept.
     * @return Predicted y value.
     */
    static double Predict(double x, double m, double b);

    /**
     * @brief Compute the mean squared error of the regression line.
     * @param x Independent variable values.
     * @param y Dependent variable values.
     * @param m Slope.
     * @param b Intercept.
     * @return Mean squared error.
     */
    static double MeanSquaredError(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   double m, double b);
};

#endif // BASIC_STATISTICS_HPP
