#include <iostream>
#include <vector>
#include <cstdlib>  // For rand()
#include <ctime>    // For time()

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1 (e.g., study hours)
    double x2;  // Feature 2 (e.g., sleep hours)
    int label;  // Binary label (0 or 1)
};

// Perceptron class
class Perceptron {
private:
    double w1;  // Weight for x1
    double w2;  // Weight for x2
    double b;   // Bias
    double learning_rate;
    int epochs;

public:
    // Constructor: Initialize weights randomly, set learning rate and epochs
    Perceptron(double lr, int ep) : learning_rate(lr), epochs(ep) {
        std::srand(std::time(0));  // Seed for random weights
        w1 = (std::rand() % 1000) / 1000.0;  // Random weight between 0 and 1
        w2 = (std::rand() % 1000) / 1000.0;
        b = (std::rand() % 1000) / 1000.0;
    }

    // Step function: Returns 1 if sum >= 0, else 0
    int step_function(double sum) const {
        return (sum >= 0) ? 1 : 0;
    }

    // Train the Perceptron on the dataset
    void train(const std::vector<DataPoint>& dataset) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& dp : dataset) {
                // Compute weighted sum
                double sum = w1 * dp.x1 + w2 * dp.x2 + b;
                int prediction = step_function(sum);
                int error = dp.label - prediction;

                // Update weights and bias if there's an error
                if (error != 0) {
                    w1 += learning_rate * error * dp.x1;
                    w2 += learning_rate * error * dp.x2;
                    b += learning_rate * error;
                }
            }
        }
    }

    // Predict the class for a new data point
    int predict(double x1, double x2) const {
        double sum = w1 * x1 + w2 * x2 + b;
        return step_function(sum);
    }

    // Display learned weights and bias
    void print_parameters() const {
        std::cout << "Learned weights: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << std::endl;
    }
};

int main() {
    // Hardcoded dataset: {study hours, sleep hours, pass/fail}
    std::vector<DataPoint> dataset = {
        {2.0, 6.0, 0},  // Fail
        {4.0, 5.0, 0},  // Fail
        {3.0, 7.0, 0},  // Fail
        {5.0, 4.0, 1},  // Pass
        {6.0, 6.0, 1},  // Pass
        {7.0, 5.0, 1}   // Pass
    };

    // Initialize and train the Perceptron
    Perceptron model(0.1, 10);  // Learning rate = 0.1, 10 epochs
    model.train(dataset);

    // Show learned parameters
    model.print_parameters();

    // User input for prediction
    std::cout << "Enter study hours and sleep hours (e.g., 5.0 6.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    // Predict and display the result
    int prediction = model.predict(x1, x2);
    std::cout << "Predicted class (0 = fail, 1 = pass): " << prediction << std::endl;

    return 0;
}