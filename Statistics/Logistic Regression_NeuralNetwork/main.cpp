#include <iostream>
#include <vector>
#include <cmath>

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1 (e.g., study hours)
    double x2;  // Feature 2 (e.g., sleep hours)
    int label;  // Binary label (0 or 1)
};

// Sigmoid function to compute probability
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Logistic Regression class
class LogisticRegression {
private:
    double w1;  // Weight for x1
    double w2;  // Weight for x2
    double b;   // Bias
    double learning_rate;
    int epochs;

public:
    // Constructor initializes weights to 0
    LogisticRegression(double lr, int ep)
        : w1(0.0), w2(0.0), b(0.0), learning_rate(lr), epochs(ep) {}

    // Train the model using batch gradient descent
    void fit(const std::vector<DataPoint>& dataset) {
        int n = dataset.size();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Gradients for weights and bias
            double dw1 = 0.0, dw2 = 0.0, db = 0.0;
            for (const auto& dp : dataset) {
                double z = w1 * dp.x1 + w2 * dp.x2 + b;
                double pred = sigmoid(z); //1 0 
                double error = pred - dp.label; // 
                dw1 += error * dp.x1;
                dw2 += error * dp.x2;
                db += error;
                printf("x1 %0.2f, x2 %0.2f,error %0.2f\n" , dp.x1,dp.x2,error);
            }
            printf("w1 = %0.2f, w2 = %0.2f, b = %0.2f\n", w1,w2,b);
            // Average gradients
            dw1 /= n;
            dw2 /= n;
            db /= n;

            // Update parameters
            w1 -= learning_rate * dw1;
            w2 -= learning_rate * dw2;
            b -= learning_rate * db;
        }
    }

    // Predict probability for new input
    double predict_probability(double x1, double x2) const {
        double z = w1 * x1 + w2 * x2 + b;
        return sigmoid(z);
    }

    // Predict class (0 or 1)
    int predict_class(double x1, double x2) const {
        return predict_probability(x1, x2) >= 0.5 ? 1 : 0;
    }

    // Display learned parameters
    void print_parameters() const {
        std::cout << "Learned weights: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << std::endl;
    }
};

int main() {
    // Hardcoded dataset: {study hours, IQ, pass/fail}
    int lowhuman = 80; 
    std::vector<DataPoint> dataset = {
        {2.0, 30, 0},  // Fail
        {3.0, 30, 0},  // Fail
        {5.0, 40, 1},  // Pass
        {7.0, 50, 1},  // Pass
        {4.0, 30, 0},  // Fail
        {6.0, 50, 1}   // Pass
    };

    // Initialize and train the model
    //LogisticRegression model(0.005, 10000);  // Learning rate = 0.05, 1000 epochs
    LogisticRegression model(0.005, 10);  // Learning rate = 0.05, 1000 epochs
    model.fit(dataset);
    //Learned weights: w1 = 1.91875, w2 = -0.146123, b = -2.888
    
    // Show learned parameters
    model.print_parameters();

    // User input for prediction
    std::cout << "Enter study hours and IQ (e.g., 5.0 100): ";
    double x1, x2;
    std::cin >> x1 >> x2;
    x2 = x2-lowhuman;
    // Predict and display results
    double prob = model.predict_probability(x1, x2);
    printf("prob = %0.2f\n", prob);
    int pred_class = model.predict_class(x1, x2);
    std::cout << "Predicted probability: " << prob << std::endl;
    std::cout << "Predicted class (0 = fail, 1 = pass): " << pred_class << std::endl;

    return 0;
}