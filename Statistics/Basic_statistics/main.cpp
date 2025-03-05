#include <iostream>
#include <vector>
#include <cmath>

// Function to calculate the mean of a vector
double mean(const std::vector<double>& data) {
    double sum = 0.0;
    for (const double& val : data) {
        sum += val;
    }
    return sum / data.size();
}

// Function to calculate the slope (m)
double slope(const std::vector<double>& x, const std::vector<double>& y, double mean_x, double mean_y) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mean_x;
        numerator += dx * (y[i] - mean_y);
        denominator += dx * dx;
    }
    return numerator / denominator;
}

// Function to calculate the y-intercept (b)
double intercept(double mean_y, double m, double mean_x) {
    return mean_y - m * mean_x;
}

// Function to predict y for a given x
double predict(double x, double m, double b) {
    return m * x + b;
}

// Function to calculate mean squared error
double mean_squared_error(const std::vector<double>& x, const std::vector<double>& y, double m, double b) {
    double sum_error = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double pred_y = m * x[i] + b;
        double error = y[i] - pred_y;
        sum_error += error * error;
    }
    return sum_error / x.size();
}

int main() {
    // Hardcoded dataset: x and y values (perfectly linear for demonstration)
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {2.0, 4.0, 6.0};

    // Calculate means
    double mean_x = mean(x);
    double mean_y = mean(y);

    // Calculate slope (m) and intercept (b)
    double m = slope(x, y, mean_x, mean_y);
    double b = intercept(mean_y, m, mean_x);

    // Display the results
    std::cout << "Slope (m): " << m << std::endl;
    std::cout << "Intercept (b): " << b << std::endl;

    // Calculate and display the mean squared error
    double mse = mean_squared_error(x, y, m, b);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // Get user input for prediction
    std::cout << "Enter a new x value to predict y: ";
    double new_x;
    std::cin >> new_x;

    // Predict and display the result
    double predicted_y = predict(new_x, m, b);
    std::cout << "Predicted y for x = " << new_x << ": " << predicted_y << std::endl;

    return 0;
}