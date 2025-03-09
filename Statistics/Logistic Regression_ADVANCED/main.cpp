#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1 (e.g., study hours)
    double x2;  // Feature 2 (e.g., IQ points above baseline)
    int label;  // Binary label (0 = fail, 1 = pass)
    
    // Constructor for easy creation
    DataPoint(double feature1, double feature2, int class_label) 
        : x1(feature1), x2(feature2), label(class_label) {}
};

// Feature normalization function
void normalize_features(std::vector<DataPoint>& dataset, double& min_x1, double& max_x1, double& min_x2, double& max_x2) {
    // Find min and max for each feature
    min_x1 = std::numeric_limits<double>::max();
    max_x1 = std::numeric_limits<double>::lowest();
    min_x2 = std::numeric_limits<double>::max();
    max_x2 = std::numeric_limits<double>::lowest();
    
    for (const auto& dp : dataset) {
        min_x1 = std::min(min_x1, dp.x1);
        max_x1 = std::max(max_x1, dp.x1);
        min_x2 = std::min(min_x2, dp.x2);
        max_x2 = std::max(max_x2, dp.x2);
    }
    
    // Normalize features to [0,1] range
    for (auto& dp : dataset) {
        dp.x1 = (dp.x1 - min_x1) / (max_x1 - min_x1);
        dp.x2 = (dp.x2 - min_x2) / (max_x2 - min_x2);
    }
    
    std::cout << "특성이 [0,1] 범위로 정규화되었습니다." << std::endl;
    std::cout << "x1 (공부 시간) 범위: [" << min_x1 << ", " << max_x1 << "]" << std::endl;
    std::cout << "x2 (조정된 IQ) 범위: [" << min_x2 << ", " << max_x2 << "]" << std::endl;
}

// Sigmoid function to compute probability (maps any value to range 0-1)
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Binary cross-entropy loss function
double binary_cross_entropy(double y_true, double y_pred) {
    // Avoid log(0) by adding small epsilon
    double epsilon = 1e-15;
    y_pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred));
    return -((y_true * std::log(y_pred)) + (1 - y_true) * std::log(1 - y_pred));
}

// Logistic Regression class with validation and visualization
class LogisticRegression {
private:
    double w1;           // Weight for feature 1
    double w2;           // Weight for feature 2
    double b;            // Bias term
    double learning_rate;
    int max_epochs;
    double l2_lambda;    // L2 regularization strength
    
    // For early stopping
    int patience;
    double best_val_loss;
    int patience_counter;
    
    // Training history
    std::vector<double> train_losses;
    std::vector<double> val_losses;

public:
    // Constructor with early stopping and regularization parameters
    LogisticRegression(double lr = 0.001, int epochs = 1000, int early_stop_patience = 100, double lambda = 0.01)
        : w1(0.0), w2(0.0), b(0.0), 
          learning_rate(lr), max_epochs(epochs),
          l2_lambda(lambda),
          patience(early_stop_patience), best_val_loss(INFINITY), patience_counter(0) {}
    
    // Compute model prediction for a single data point
    double predict_probability(double x1, double x2) const {
        double z = w1 * x1 + w2 * x2 + b;  // Linear combination of inputs
        return sigmoid(z);                 // Apply sigmoid to get probability
    }
    
    // Predict class (0 or 1) based on 0.5 threshold
    int predict_class(double x1, double x2) const {
        return predict_probability(x1, x2) >= 0.5 ? 1 : 0;
    }
    
    // Calculate accuracy on a dataset
    double calculate_accuracy(const std::vector<DataPoint>& dataset) const {
        int correct = 0;
        for (const auto& dp : dataset) {
            int prediction = predict_class(dp.x1, dp.x2);
            if (prediction == dp.label) {
                correct++;
            }
        }
        return static_cast<double>(correct) / dataset.size() * 100.0;
    }
    
    // Calculate average loss on a dataset (with L2 regularization)
    double calculate_loss(const std::vector<DataPoint>& dataset) const {
        double total_loss = 0.0;
        for (const auto& dp : dataset) {
            double pred = predict_probability(dp.x1, dp.x2);
            total_loss += binary_cross_entropy(dp.label, pred);
        }
        // Add L2 regularization term
        double l2_term = 0.5 * l2_lambda * (w1*w1 + w2*w2);
        return (total_loss / dataset.size()) + l2_term;
    }
    
    // Train the model using batch gradient descent with validation
    void fit(const std::vector<DataPoint>& train_data, 
             const std::vector<DataPoint>& val_data,
             bool verbose = true) {
        
        int n = train_data.size();
        
        // Initialize best parameters for early stopping
        double best_w1 = w1;
        double best_w2 = w2;
        double best_b = b;
        
        // Clear history
        train_losses.clear();
        val_losses.clear();
        
        // Print header for training progress
        if (verbose) {
            std::cout << std::setw(7) << "Epoch" 
                      << std::setw(15) << "Train Loss" 
                      << std::setw(15) << "Val Loss" 
                      << std::setw(15) << "Train Acc(%)" 
                      << std::setw(15) << "Val Acc(%)" 
                      << std::setw(22) << "Parameters (w1,w2,b)" << std::endl;
            std::cout << std::string(89, '-') << std::endl;
        }
        
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            // Gradients for weights and bias
            double dw1 = 0.0, dw2 = 0.0, db = 0.0;
            
            // Compute gradients for each training example
            for (const auto& dp : train_data) {
                double pred = predict_probability(dp.x1, dp.x2);
                double error = pred - dp.label;  // Prediction error
                
                // Accumulate gradients
                dw1 += error * dp.x1;
                dw2 += error * dp.x2;
                db += error;
            }
            
            // Average gradients and add L2 regularization
            dw1 = dw1/n + l2_lambda * w1;
            dw2 = dw2/n + l2_lambda * w2;
            db /= n;  // No regularization for bias
            
            // Update parameters using gradient descent
            w1 -= learning_rate * dw1;
            w2 -= learning_rate * dw2;
            b -= learning_rate * db;
            
            // Calculate losses
            double train_loss = calculate_loss(train_data);
            double val_loss = calculate_loss(val_data);
            
            // Store losses for plotting
            train_losses.push_back(train_loss);
            val_losses.push_back(val_loss);
            
            // Calculate accuracies
            double train_acc = calculate_accuracy(train_data);
            double val_acc = calculate_accuracy(val_data);
            
            // Print progress
            if (verbose && (epoch % 100 == 0 || epoch == max_epochs - 1)) {
                std::cout << std::setw(7) << epoch 
                          << std::setw(15) << std::fixed << std::setprecision(6) << train_loss
                          << std::setw(15) << val_loss
                          << std::setw(15) << std::setprecision(2) << train_acc
                          << std::setw(15) << val_acc
                          << "   (" << std::setprecision(4) << w1 << ", " 
                          << w2 << ", " << b << ")" << std::endl;
            }
            
            // Early stopping check
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                
                // Save best parameters
                best_w1 = w1;
                best_w2 = w2;
                best_b = b;
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    if (verbose) {
                        std::cout << "Early stopping at epoch " << epoch 
                                  << " (no improvement for " << patience << " epochs)" << std::endl;
                    }
                    break;
                }
            }
        }
        
        // Restore best parameters
        w1 = best_w1;
        w2 = best_w2;
        b = best_b;
        
        if (verbose) {
            std::cout << "\nFinal parameters:" << std::endl;
            print_parameters();
            
            // Display decision boundary equation
            std::cout << "Decision boundary equation: " << w1 << " * x1 + " 
                      << w2 << " * x2 + " << b << " = 0" << std::endl;
            
            // Interpretation of the model
            std::cout << "\nInterpretation:" << std::endl;
            if (w1 > 0) {
                std::cout << "- Higher values of feature 1 (study hours) increase the probability of passing" << std::endl;
            } else {
                std::cout << "- Higher values of feature 1 (study hours) decrease the probability of passing" << std::endl;
            }
            
            if (w2 > 0) {
                std::cout << "- Higher values of feature 2 (adjusted IQ) increase the probability of passing" << std::endl;
            } else {
                std::cout << "- Higher values of feature 2 (adjusted IQ) decrease the probability of passing" << std::endl;
            }
            
            // // Example prediction interpretation
            // double threshold_x1 = -b / w1;  // When x2 = 0
            // double threshold_x2 = -b / w2;  // When x1 = 0
            
            // std::cout << "- With no adjusted IQ advantage (x2=0), you need to study about " 
            //           << std::max(0.0, threshold_x1) << " hours to have a 50% chance of passing" << std::endl;
            // std::cout << "- With no study (x1=0), you need an adjusted IQ advantage of about " 
            //           << std::max(0.0, threshold_x2) << " points to have a 50% chance of passing" << std::endl;
        }
    }
    
    // Display learned parameters
    void print_parameters() const {
        std::cout << "w1 (study hours) = " << w1 << "\n"
                  << "w2 (adjusted IQ) = " << w2 << "\n"
                  << "b (bias) = " << b << std::endl;
    }
    
    // Get the training history
    std::pair<std::vector<double>, std::vector<double>> get_history() const {
        return {train_losses, val_losses};
    }
};

// Function to split data into training and validation sets
std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
train_test_split(const std::vector<DataPoint>& dataset, double test_size = 0.2) {
    // Create a copy of the dataset
    std::vector<DataPoint> data = dataset;
    
    // Shuffle the data with fixed seed for reproducibility
    std::mt19937 g(42);  // Fixed seed
    std::shuffle(data.begin(), data.end(), g);
    
    // Calculate split point
    size_t test_samples = static_cast<size_t>(data.size() * test_size);
    size_t train_samples = data.size() - test_samples;
    
    // Split the data
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_samples);
    std::vector<DataPoint> test_data(data.begin() + train_samples, data.end());
    
    return {train_data, test_data};
}

int main() {
    // Define baseline IQ (to make the model more interpretable)
    int baseline_iq = 80;
    
    // Create dataset: {study hours, IQ points above baseline, pass/fail}
    std::vector<DataPoint> dataset = {
        // 원본 데이터
        {2.0, 30, 0},  // 2 hours of study, IQ 110 (110-80=30), Fail
        {3.0, 30, 0},  // 3 hours of study, IQ 110 (110-80=30), Fail
        {5.0, 40, 1},  // 5 hours of study, IQ 120 (120-80=40), Pass
        {7.0, 50, 1},  // 7 hours of study, IQ 130 (130-80=50), Pass
        {4.0, 30, 0},  // 4 hours of study, IQ 110 (110-80=30), Fail
        {6.0, 50, 1},  // 6 hours of study, IQ 130 (130-80=50), Pass
        {3.5, 35, 0},  // Adding more data points for better training
        {4.5, 40, 1},
        {6.5, 45, 1},
        {2.5, 25, 0},
        {5.5, 45, 1},
        {3.0, 40, 0},
        
        // 추가 데이터 (기존 패턴 유지)
        {1.5, 25, 0},  // 낮은 공부시간, 낮은 IQ, 불합격
        {2.2, 35, 0},  // 낮은 공부시간, 중간 IQ, 불합격
        {2.7, 28, 0},  // 낮은 공부시간, 낮은 IQ, 불합격
        {3.2, 32, 0},  // 낮은 공부시간, 중간 IQ, 불합격
        {3.8, 38, 0},  // 중간 공부시간, 중간 IQ, 불합격
        {4.2, 35, 0},  // 중간 공부시간, 중간 IQ, 불합격
        {4.8, 42, 1},  // 중간 공부시간, 높은 IQ, 합격
        {5.2, 38, 1},  // 높은 공부시간, 중간 IQ, 합격
        {5.8, 45, 1},  // 높은 공부시간, 높은 IQ, 합격
        {6.2, 40, 1},  // 높은 공부시간, 높은 IQ, 합격
        {6.8, 48, 1},  // 높은 공부시간, 높은 IQ, 합격
        {7.5, 45, 1},  // 높은 공부시간, 높은 IQ, 합격
        
        // 경계 케이스 (약간의 노이즈 포함)
        {4.5, 35, 0},  // 경계선상, 불합격
        {4.8, 36, 1},  // 경계선상, 합격
        {5.0, 35, 0},  // 경계선상, 불합격
        {5.2, 36, 1},  // 경계선상, 합격
        {4.0, 42, 1},  // 낮은 공부시간, 높은 IQ, 합격
        {6.0, 32, 1},  // 높은 공부시간, 낮은 IQ, 합격
        {3.8, 45, 1},  // 낮은 공부시간, 높은 IQ, 합격
        {5.5, 30, 0},  // 높은 공부시간, 낮은 IQ, 불합격
        
        // 다양한 추가 데이터
        {1.0, 20, 0},  // 매우 낮은 공부시간, 매우 낮은 IQ, 불합격
        {8.0, 55, 1},  // 매우 높은 공부시간, 매우 높은 IQ, 합격
        {2.0, 50, 0},  // 낮은 공부시간, 높은 IQ, 불합격 
        {7.0, 25, 1},  // 높은 공부시간, 낮은 IQ, 합격
        {3.5, 45, 1},  // 중간 공부시간, 높은 IQ, 합격
        {6.5, 35, 1},  // 높은 공부시간, 중간 IQ, 합격
        {4.2, 30, 0},  // 중간 공부시간, 낮은 IQ, 불합격
        {5.5, 40, 1},  // 높은 공부시간, 중간 IQ, 합격
        
        // 극단적 케이스
        {9.0, 60, 1},  // 매우 높은 공부시간, 매우 높은 IQ, 합격
        {0.5, 15, 0},  // 매우 낮은 공부시간, 매우 낮은 IQ, 불합격
        {8.5, 30, 1},  // 매우 높은 공부시간, 낮은 IQ, 합격
        {1.5, 55, 0},  // 매우 낮은 공부시간, 매우 높은 IQ, 불합격 
        
        // 분류 경계를 더 명확히 하는 추가 데이터
        {4.7, 38, 0},  // 경계선상, 불합격
        {4.9, 39, 1},  // 경계선상, 합격
        {5.1, 37, 1},  // 경계선상, 합격 
        {4.6, 41, 1},  // 경계선상, 합격
        {5.3, 34, 0},  // 경계선상, 불합격
    };
    
    // 특성 정규화 - 미리 저장할 변수들
    double min_x1, max_x1, min_x2, max_x2;
    
    // 데이터셋 정규화
    normalize_features(dataset, min_x1, max_x1, min_x2, max_x2);
    
    // Split data into training and validation sets
    auto [train_data, val_data] = train_test_split(dataset, 0.25);
    
    std::cout << "Training with " << train_data.size() << " examples" << std::endl;
    std::cout << "Validating with " << val_data.size() << " examples" << std::endl;
    
    // Initialize logistic regression model with improved parameters
    LogisticRegression model(0.01, 10000, 1000, 0.01);  // Learning rate, max epochs, patience, L2 lambda
    
    // Train the model
    model.fit(train_data, val_data);
    
    // Ask user for prediction inputs
    std::cout << "\n=== Prediction Mode ===" << std::endl;
    std::cout << "Enter study hours and IQ (e.g., 5.0 120): ";
    double study_hours, iq;
    std::cin >> study_hours >> iq;
    
    // Adjust IQ to be relative to baseline
    double adjusted_iq = iq - baseline_iq;
    
    // Normalize user inputs using the same scaling as training data
    double normalized_study = (study_hours - min_x1) / (max_x1 - min_x1);
    double normalized_iq = (adjusted_iq - min_x2) / (max_x2 - min_x2);
    
    // Make prediction with normalized inputs
    double prob = model.predict_probability(normalized_study, normalized_iq);
    int predicted_class = model.predict_class(normalized_study, normalized_iq);
    
    // Display results with interpretation
    std::cout << "\nPrediction Results:" << std::endl;
    std::cout << "Study Hours: " << study_hours << " (normalized: " << normalized_study << ")" << std::endl;
    std::cout << "IQ: " << iq << " (adjusted: " << adjusted_iq << ", normalized: " << normalized_iq << ")" << std::endl;
    std::cout << "Probability of passing: " << std::fixed << std::setprecision(2) << prob * 100 << "%" << std::endl;
    std::cout << "Predicted outcome: " << (predicted_class ? "PASS" : "FAIL") << std::endl;
    
    return 0;
}