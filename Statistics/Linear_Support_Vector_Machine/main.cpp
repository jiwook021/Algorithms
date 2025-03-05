#include <iostream>
#include <vector>
#include <cmath>

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1
    double x2;  // Feature 2
    int label;  // Binary label (-1 or 1 for SVM)
};

// Linear SVM class
class LinearSVM {
private:
    double w1;  // Weight for feature x1
    double w2;  // Weight for feature x2
    double b;   // Bias term
    double learning_rate;
    double regularization;  // C parameter for regularization strength
    int epochs;

public:
    // Constructor: Initialize weights and bias to 0, set hyperparameters
    LinearSVM(double lr, double reg, int ep)
        : w1(0.0), w2(0.0), b(0.0), learning_rate(lr), regularization(reg), epochs(ep) {}

    // Decision function: f(x) = w1*x1 + w2*x2 + b
    double decision_function(double x1, double x2) const {
        return w1 * x1 + w2 * x2 + b;
    }

    // Predict the class: 1 if decision function >= 0, else -1
    int predict(double x1, double x2) const {
        return decision_function(x1, x2) >= 0 ? 1 : -1;
    }

    // Hinge loss: max(0, 1 - y * f(x))
    double hinge_loss(const DataPoint& dp) const {
        double margin = dp.label * decision_function(dp.x1, dp.x2);
        return std::max(0.0, 1.0 - margin);
    }

    // Train the SVM using gradient descent to minimize hinge loss
    void train(const std::vector<DataPoint>& dataset) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& dp : dataset) {
                double margin = dp.label * decision_function(dp.x1, dp.x2);
                if (margin < 1) {  // Point is misclassified or within the margin
                    // Subgradient updates for weights and bias
                    double grad_w1 = -dp.label * dp.x1 + (w1 / regularization);
                    double grad_w2 = -dp.label * dp.x2 + (w2 / regularization);
                    double grad_b = -dp.label;
                    // Update parameters
                    w1 -= learning_rate * grad_w1;
                    w2 -= learning_rate * grad_w2;
                    b -= learning_rate * grad_b;
                } else {  // Point is correctly classified outside the margin
                    // Apply regularization to weights only
                    w1 -= learning_rate * (w1 / regularization);
                    w2 -= learning_rate * (w2 / regularization);
                    // Bias is not regularized
                }
            }
        }
    }

    // Display learned parameters
    void print_parameters() const {
        std::cout << "Learned weights: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << std::endl;
    }
};

int main() {
    // Hardcoded dataset: {x1, x2, label}
    std::vector<DataPoint> dataset = {
        {2.0, 3.0, -1},  // Class -1
        {3.0, 3.0, -1},
        {3.0, 4.0, -1},
        {5.0, 5.0, 1},   // Class 1
        {6.0, 5.0, 1},
        {7.0, 6.0, 1}
    };

    // Initialize SVM with learning rate = 0.01, regularization = 1.0, 100 epochs
    LinearSVM model(0.01, 1.0, 100);
    model.train(dataset);

    // Show the learned parameters
    model.print_parameters();

    // User input for prediction
    std::cout << "Enter x1 and x2 (e.g., 4.0 4.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    // Predict and display the result
    int prediction = model.predict(x1, x2);
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}