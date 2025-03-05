#include <iostream>
#include <vector>

// Structure to hold a data point with one feature and one output
struct DataPoint {
    double x;  // Feature
    double y;  // Output
};

// Class to handle Linear Regression
class LinearRegression {
private:
    double m;  // Slope
    double b;  // Intercept

public:
    LinearRegression() : m(0.0), b(0.0) {}

    // Train the model using the least squares method
    void fit(const std::vector<DataPoint>& dataset) {
        double x_sum = 0.0, y_sum = 0.0, xy_sum = 0.0, x2_sum = 0.0;
        int n = dataset.size();

        // Calculate sums for the least squares formula
        for (const auto& dp : dataset) {
            x_sum += dp.x;
            y_sum += dp.y;
            xy_sum += dp.x * dp.y;
            x2_sum += dp.x * dp.x;
        }

        // Calculate slope (m) and intercept (b)
        m = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
        b = (y_sum - m * x_sum) / n;
    }

    // Predict the output for a new input
    double predict(double x) const {
        return m * x + b;
    }

    // Getters for slope and intercept (for display purposes)
    double get_slope() const { return m; }
    double get_intercept() const { return b; }
};

int main() {
    // Hardcoded dataset: pairs of (x, y)
    std::vector<DataPoint> dataset = {
        {1.0, 2.1},
        {2.0, 3.8},
        {3.0, 5.2},
        {4.0, 7.0},
        {5.0, 8.9}
    };

    // Create and train the Linear Regression model
    LinearRegression model;
    model.fit(dataset);

    // Display the learned parameters
    std::cout << "Learned model: y = " << model.get_slope() << "x + " << model.get_intercept() << std::endl;

    // Get user input for a new x value
    std::cout << "Enter an x value to predict y: ";
    double x;
    std::cin >> x;

    // Predict and display the result
    double y_pred = model.predict(x);
    std::cout << "Predicted y for x = " << x << ": " << y_pred << std::endl;

    return 0;
}