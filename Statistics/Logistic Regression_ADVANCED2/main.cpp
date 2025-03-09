#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <numeric>

// Structure for a data point with features and a binary label
struct DataPoint {
    double x1;  // Feature 1 (e.g., study hours)
    double x2;  // Feature 2 (e.g., IQ points above baseline)
    double x3 = 0.0;  // Interaction term (x1 * x2)
    double x4 = 0.0;  // Polynomial feature (x1^2)
    double x5 = 0.0;  // Polynomial feature (x2^2)
    int label;  // Binary label (0 = fail, 1 = pass)
    
    // Constructor for easy creation
    DataPoint(double feature1, double feature2, int class_label) 
        : x1(feature1), x2(feature2), label(class_label) {}
};

// Feature engineering: add interaction and polynomial features
void add_engineered_features(std::vector<DataPoint>& dataset) {
    for (auto& dp : dataset) {
        // Add interaction term (x1 * x2)
        dp.x3 = dp.x1 * dp.x2;
        
        // Add polynomial features
        dp.x4 = dp.x1 * dp.x1;  // x1 squared
        dp.x5 = dp.x2 * dp.x2;  // x2 squared
    }
}

// Feature normalization function
void normalize_features(std::vector<DataPoint>& dataset, 
                        std::vector<double>& min_values, 
                        std::vector<double>& max_values) {
    // Initialize with appropriate size (5 features)
    min_values.resize(5, std::numeric_limits<double>::max());
    max_values.resize(5, std::numeric_limits<double>::lowest());
    
    // Find min and max for each feature
    for (const auto& dp : dataset) {
        min_values[0] = std::min(min_values[0], dp.x1);
        max_values[0] = std::max(max_values[0], dp.x1);
        min_values[1] = std::min(min_values[1], dp.x2);
        max_values[1] = std::max(max_values[1], dp.x2);
    }
    
    // Add engineered features before normalizing them
    add_engineered_features(dataset);
    
    // Find min/max for engineered features
    for (const auto& dp : dataset) {
        min_values[2] = std::min(min_values[2], dp.x3);
        max_values[2] = std::max(max_values[2], dp.x3);
        min_values[3] = std::min(min_values[3], dp.x4);
        max_values[3] = std::max(max_values[3], dp.x4);
        min_values[4] = std::min(min_values[4], dp.x5);
        max_values[4] = std::max(max_values[4], dp.x5);
    }
    
    // Normalize features to [0,1] range
    for (auto& dp : dataset) {
        dp.x1 = (dp.x1 - min_values[0]) / (max_values[0] - min_values[0]);
        dp.x2 = (dp.x2 - min_values[1]) / (max_values[1] - min_values[1]);
        dp.x3 = (dp.x3 - min_values[2]) / (max_values[2] - min_values[2]);
        dp.x4 = (dp.x4 - min_values[3]) / (max_values[3] - min_values[3]);
        dp.x5 = (dp.x5 - min_values[4]) / (max_values[4] - min_values[4]);
    }
    
    std::cout << "특성이 [0,1] 범위로 정규화되었습니다." << std::endl;
    std::cout << "x1 (공부 시간) 범위: [" << min_values[0] << ", " << max_values[0] << "]" << std::endl;
    std::cout << "x2 (조정된 IQ) 범위: [" << min_values[1] << ", " << max_values[1] << "]" << std::endl;
}

// Normalize a single data point using existing min/max values
void normalize_datapoint(DataPoint& dp, const std::vector<double>& min_values, 
                        const std::vector<double>& max_values) {
    // Add engineered features first
    dp.x3 = dp.x1 * dp.x2;
    dp.x4 = dp.x1 * dp.x1;
    dp.x5 = dp.x2 * dp.x2;
    
    // Normalize
    dp.x1 = (dp.x1 - min_values[0]) / (max_values[0] - min_values[0]);
    dp.x2 = (dp.x2 - min_values[1]) / (max_values[1] - min_values[1]);
    dp.x3 = (dp.x3 - min_values[2]) / (max_values[2] - min_values[2]);
    dp.x4 = (dp.x4 - min_values[3]) / (max_values[3] - min_values[3]);
    dp.x5 = (dp.x5 - min_values[4]) / (max_values[4] - min_values[4]);
}

// Sigmoid function to compute probability (maps any value to range 0-1)
double sigmoid(double z) {
    // Clip z to avoid overflow
    z = std::max(-20.0, std::min(z, 20.0));
    return 1.0 / (1.0 + std::exp(-z));
}

// Binary cross-entropy loss function
double binary_cross_entropy(double y_true, double y_pred) {
    // Avoid log(0) by adding small epsilon
    double epsilon = 1e-15;
    y_pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred));
    return -((y_true * std::log(y_pred)) + (1 - y_true) * std::log(1 - y_pred));
}

// Structure for model evaluation metrics
struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double auc;  // Area Under ROC Curve
};

// Enhanced Logistic Regression class with validation and advanced features
class LogisticRegression {
private:
    // Model parameters
    std::vector<double> weights;  // Weights for all features
    double b;                    // Bias term
    
    // Hyperparameters
    double initial_learning_rate;
    double learning_rate;
    double decay_rate;
    int max_epochs;
    double l2_lambda;    // L2 regularization strength
    bool use_momentum;
    double momentum;     // Momentum coefficient
    bool use_all_features; // Whether to use engineered features
    
    // Momentum variables
    std::vector<double> prev_dw;
    double prev_db;
    
    // For early stopping
    int patience;
    double best_val_loss;
    int patience_counter;
    
    // Training history
    std::vector<double> train_losses;
    std::vector<double> val_losses;
    std::vector<double> train_accuracies;
    std::vector<double> val_accuracies;
    
    // Update learning rate based on schedule
    void update_learning_rate(int epoch) {
        // Simple decay
        learning_rate = initial_learning_rate / (1.0 + decay_rate * epoch);
    }

public:
    // Constructor with advanced options
    LogisticRegression(double lr = 0.01, int epochs = 1000, int early_stop_patience = 100, 
                      double lambda = 0.01, bool use_momentum = true, double momentum_val = 0.9,
                      bool use_all_features = false, double lr_decay = 0.0001)
        : b(0.0), 
          initial_learning_rate(lr), learning_rate(lr), decay_rate(lr_decay),
          max_epochs(epochs), l2_lambda(lambda),
          use_momentum(use_momentum), momentum(momentum_val),
          use_all_features(use_all_features),
          patience(early_stop_patience), best_val_loss(INFINITY), patience_counter(0) {
        
        // Initialize weights depending on feature usage
        int feature_count = use_all_features ? 5 : 2;
        weights.resize(feature_count, 0.0);
        
        // Initialize momentum variables
        prev_dw.resize(feature_count, 0.0);
        prev_db = 0.0;
    }
    
    // Compute model prediction for a single data point
    double predict_probability(const DataPoint& dp) const {
        double z = b;  // Start with bias
        
        // Add weighted features
        z += weights[0] * dp.x1 + weights[1] * dp.x2;
        
        // Add engineered features if used
        if (use_all_features && weights.size() >= 5) {
            z += weights[2] * dp.x3;  // Interaction
            z += weights[3] * dp.x4;  // x1^2
            z += weights[4] * dp.x5;  // x2^2
        }
        
        return sigmoid(z);
    }
    
    // Alternative version for direct input when using only basic features
    double predict_probability(double x1, double x2) const {
        double z = weights[0] * x1 + weights[1] * x2 + b;
        return sigmoid(z);
    }
    
    // Predict class (0 or 1) based on 0.5 threshold
    int predict_class(const DataPoint& dp) const {
        return predict_probability(dp) >= 0.5 ? 1 : 0;
    }
    
    // Alternative version for direct input
    int predict_class(double x1, double x2) const {
        return predict_probability(x1, x2) >= 0.5 ? 1 : 0;
    }
    
    // Calculate accuracy on a dataset
    double calculate_accuracy(const std::vector<DataPoint>& dataset) const {
        int correct = 0;
        for (const auto& dp : dataset) {
            int prediction = predict_class(dp);
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
            double pred = predict_probability(dp);
            total_loss += binary_cross_entropy(dp.label, pred);
        }
        
        // Add L2 regularization term
        double l2_term = 0.0;
        for (const auto& w : weights) {
            l2_term += w * w;
        }
        l2_term = 0.5 * l2_lambda * l2_term;
        
        return (total_loss / dataset.size()) + l2_term;
    }
    
    // Comprehensive evaluation metrics
    Metrics evaluate(const std::vector<DataPoint>& dataset) const {
        int true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
        
        for (const auto& dp : dataset) {
            int prediction = predict_class(dp);
            if (prediction == 1 && dp.label == 1) true_positive++;
            else if (prediction == 1 && dp.label == 0) false_positive++;
            else if (prediction == 0 && dp.label == 0) true_negative++;
            else if (prediction == 0 && dp.label == 1) false_negative++;
        }
        
        double accuracy = static_cast<double>(true_positive + true_negative) / dataset.size();
        
        // Avoid division by zero
        double precision = (true_positive + false_positive > 0) ? 
                          static_cast<double>(true_positive) / (true_positive + false_positive) : 0.0;
        
        double recall = (true_positive + false_negative > 0) ?
                       static_cast<double>(true_positive) / (true_positive + false_negative) : 0.0;
        
        double f1_score = (precision + recall > 0) ?
                         2.0 * (precision * recall) / (precision + recall) : 0.0;
        
        // Simplified AUC calculation (for binary classification)
        double auc = (recall + (static_cast<double>(true_negative) / 
                               (true_negative + false_positive))) / 2.0;
        
        return {accuracy * 100.0, precision * 100.0, recall * 100.0, f1_score * 100.0, auc * 100.0};
    }
    
    // K-fold cross-validation
    Metrics cross_validate(const std::vector<DataPoint>& dataset, int k_folds = 5) {
        std::vector<DataPoint> data_copy = dataset;
        std::mt19937 g(42);
        std::shuffle(data_copy.begin(), data_copy.end(), g);
        
        size_t fold_size = data_copy.size() / k_folds;
        std::vector<double> fold_metrics(5, 0.0);  // acc, prec, recall, f1, auc
        
        std::cout << "\nPerforming " << k_folds << "-fold cross-validation..." << std::endl;
        
        for (int fold = 0; fold < k_folds; fold++) {
            // Split data into training and validation
            size_t start_idx = fold * fold_size;
            size_t end_idx = (fold == k_folds - 1) ? data_copy.size() : start_idx + fold_size;
            
            std::vector<DataPoint> val_fold(data_copy.begin() + start_idx, 
                                           data_copy.begin() + end_idx);
            
            std::vector<DataPoint> train_fold;
            train_fold.insert(train_fold.end(), data_copy.begin(), data_copy.begin() + start_idx);
            train_fold.insert(train_fold.end(), data_copy.begin() + end_idx, data_copy.end());
            
            // Create a new model with the same hyperparameters
            LogisticRegression fold_model(initial_learning_rate, max_epochs, patience, 
                                         l2_lambda, use_momentum, momentum, use_all_features);
            
            // Train and evaluate
            fold_model.fit(train_fold, val_fold, false);
            Metrics fold_result = fold_model.evaluate(val_fold);
            
            // Accumulate metrics
            fold_metrics[0] += fold_result.accuracy;
            fold_metrics[1] += fold_result.precision;
            fold_metrics[2] += fold_result.recall;
            fold_metrics[3] += fold_result.f1_score;
            fold_metrics[4] += fold_result.auc;
            
            std::cout << "Fold " << fold + 1 << " - Accuracy: " << fold_result.accuracy 
                      << "%, F1: " << fold_result.f1_score << "%" << std::endl;
        }
        
        // Average the metrics
        Metrics avg_metrics;
        avg_metrics.accuracy = fold_metrics[0] / k_folds;
        avg_metrics.precision = fold_metrics[1] / k_folds;
        avg_metrics.recall = fold_metrics[2] / k_folds;
        avg_metrics.f1_score = fold_metrics[3] / k_folds;
        avg_metrics.auc = fold_metrics[4] / k_folds;
        
        std::cout << "Cross-validation results:" << std::endl;
        std::cout << "  Average Accuracy: " << avg_metrics.accuracy << "%" << std::endl;
        std::cout << "  Average Precision: " << avg_metrics.precision << "%" << std::endl;
        std::cout << "  Average Recall: " << avg_metrics.recall << "%" << std::endl;
        std::cout << "  Average F1 Score: " << avg_metrics.f1_score << "%" << std::endl;
        std::cout << "  Average AUC: " << avg_metrics.auc << "%" << std::endl;
        
        return avg_metrics;
    }
    
    // Train the model using batch gradient descent with validation
    void fit(const std::vector<DataPoint>& train_data, 
             const std::vector<DataPoint>& val_data,
             bool verbose = true) {
        
        int n = train_data.size();
        
        // Initialize best parameters for early stopping
        std::vector<double> best_weights = weights;
        double best_b = b;
        
        // Clear history
        train_losses.clear();
        val_losses.clear();
        train_accuracies.clear();
        val_accuracies.clear();
        
        // Print header for training progress
        if (verbose) {
            std::cout << "\nTraining model with " << weights.size() << " features..." << std::endl;
            std::cout << std::setw(7) << "Epoch" 
                      << std::setw(15) << "Train Loss" 
                      << std::setw(15) << "Val Loss" 
                      << std::setw(15) << "Train Acc(%)" 
                      << std::setw(15) << "Val Acc(%)" 
                      << std::setw(22) << "Learning Rate" << std::endl;
            std::cout << std::string(89, '-') << std::endl;
        }
        
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            // Update learning rate if decay is enabled
            if (decay_rate > 0) {
                update_learning_rate(epoch);
            }
            
            // Gradients for weights and bias
            std::vector<double> dw(weights.size(), 0.0);
            double db = 0.0;
            
            // Compute gradients for each training example
            for (const auto& dp : train_data) {
                double pred = predict_probability(dp);
                double error = pred - dp.label;  // Prediction error
                
                // Accumulate gradients for basic features
                dw[0] += error * dp.x1;
                dw[1] += error * dp.x2;
                
                // Add gradients for engineered features if used
                if (use_all_features && weights.size() >= 5) {
                    dw[2] += error * dp.x3;
                    dw[3] += error * dp.x4;
                    dw[4] += error * dp.x5;
                }
                
                db += error;
            }
            
            // Average gradients and add L2 regularization
            for (size_t i = 0; i < weights.size(); ++i) {
                dw[i] = dw[i]/n + l2_lambda * weights[i];
            }
            db /= n;  // No regularization for bias
            
            // Apply momentum if enabled
            if (use_momentum) {
                for (size_t i = 0; i < weights.size(); ++i) {
                    dw[i] = momentum * prev_dw[i] + (1.0 - momentum) * dw[i];
                    prev_dw[i] = dw[i];
                }
                db = momentum * prev_db + (1.0 - momentum) * db;
                prev_db = db;
            }
            
            // Update parameters using gradient descent
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= learning_rate * dw[i];
            }
            b -= learning_rate * db;
            
            // Calculate losses and accuracies
            double train_loss = calculate_loss(train_data);
            double val_loss = calculate_loss(val_data);
            double train_acc = calculate_accuracy(train_data);
            double val_acc = calculate_accuracy(val_data);
            
            // Store history
            train_losses.push_back(train_loss);
            val_losses.push_back(val_loss);
            train_accuracies.push_back(train_acc);
            val_accuracies.push_back(val_acc);
            
            // Print progress
            if (verbose && (epoch % 100 == 0 || epoch == max_epochs - 1)) {
                std::cout << std::setw(7) << epoch 
                          << std::setw(15) << std::fixed << std::setprecision(6) << train_loss
                          << std::setw(15) << val_loss
                          << std::setw(15) << std::setprecision(2) << train_acc
                          << std::setw(15) << val_acc
                          << std::setw(15) << std::setprecision(6) << learning_rate << std::endl;
            }
            
            // Early stopping check
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                
                // Save best parameters
                best_weights = weights;
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
        weights = best_weights;
        b = best_b;
        
        if (verbose) {
            std::cout << "\nFinal parameters:" << std::endl;
            print_parameters();
            
            // Display decision boundary equation for basic features
            std::cout << "Decision boundary equation: " << weights[0] << " * x1 + " 
                      << weights[1] << " * x2";
            
            if (use_all_features && weights.size() >= 5) {
                std::cout << " + " << weights[2] << " * x1*x2";
                std::cout << " + " << weights[3] << " * x1^2";
                std::cout << " + " << weights[4] << " * x2^2";
            }
            
            std::cout << " + " << b << " = 0" << std::endl;
            
            // Interpretation of the model
            std::cout << "\nInterpretation:" << std::endl;
            std::cout << "- Study hours (x1) impact: " << (weights[0] > 0 ? "Positive" : "Negative") 
                      << " (" << weights[0] << ")" << std::endl;
            std::cout << "- IQ (x2) impact: " << (weights[1] > 0 ? "Positive" : "Negative") 
                      << " (" << weights[1] << ")" << std::endl;
            
            if (use_all_features && weights.size() >= 5) {
                std::cout << "- Interaction term (x1*x2) impact: " 
                          << (weights[2] > 0 ? "Positive" : "Negative") 
                          << " (" << weights[2] << ")" << std::endl;
                std::cout << "- Study hours squared (x1^2) impact: " 
                          << (weights[3] > 0 ? "Positive" : "Negative") 
                          << " (" << weights[3] << ")" << std::endl;
                std::cout << "- IQ squared (x2^2) impact: " 
                          << (weights[4] > 0 ? "Positive" : "Negative") 
                          << " (" << weights[4] << ")" << std::endl;
            }
        }
    }
    
    // Display learned parameters
    void print_parameters() const {
        std::cout << "w1 (study hours) = " << weights[0] << std::endl;
        std::cout << "w2 (adjusted IQ) = " << weights[1] << std::endl;
        
        if (use_all_features && weights.size() >= 5) {
            std::cout << "w3 (interaction) = " << weights[2] << std::endl;
            std::cout << "w4 (study^2) = " << weights[3] << std::endl;
            std::cout << "w5 (IQ^2) = " << weights[4] << std::endl;
        }
        
        std::cout << "b (bias) = " << b << std::endl;
    }
    
    // Get the training history
    std::pair<std::vector<double>, std::vector<double>> get_loss_history() const {
        return {train_losses, val_losses};
    }
    
    std::pair<std::vector<double>, std::vector<double>> get_accuracy_history() const {
        return {train_accuracies, val_accuracies};
    }
    
    // Save model to file
    void save_model(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return;
        }
        
        // Save number of features and regularization parameter
        file << weights.size() << " " << l2_lambda << std::endl;
        
        // Save weights
        for (const auto& w : weights) {
            file << w << " ";
        }
        file << std::endl;
        
        // Save bias
        file << b << std::endl;
        
        file.close();
        std::cout << "Model saved to: " << filename << std::endl;
    }
    
    // Load model from file
    bool load_model(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
            return false;
        }
        
        // Load number of features and lambda
        int feature_count;
        file >> feature_count >> l2_lambda;
        
        // Resize weights vector
        weights.resize(feature_count);
        
        // Load weights
        for (int i = 0; i < feature_count; ++i) {
            file >> weights[i];
        }
        
        // Load bias
        file >> b;
        
        file.close();
        std::cout << "Model loaded from: " << filename << std::endl;
        
        // Update use_all_features flag
        use_all_features = (feature_count >= 5);
        
        return true;
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

// Hyperparameter tuning function
LogisticRegression tune_hyperparameters(
    const std::vector<DataPoint>& train_data,
    const std::vector<DataPoint>& val_data,
    bool use_all_features = false) {
    
    std::cout << "\nPerforming hyperparameter tuning..." << std::endl;
    
    // Define hyperparameter grid
    std::vector<double> learning_rates = {0.001, 0.01, 0.1};
    std::vector<double> lambda_values = {0.0001, 0.001, 0.01, 0.1};
    std::vector<bool> momentum_options = {true, false};
    
    double best_val_acc = 0.0;
    double best_lr = 0.0, best_lambda = 0.0;
    bool best_momentum = true;
    
    // Grid search
    for (double lr : learning_rates) {
        for (double lambda : lambda_values) {
            for (bool use_momentum : momentum_options) {
                std::cout << "Testing LR=" << lr << ", Lambda=" << lambda 
                          << ", Momentum=" << (use_momentum ? "yes" : "no") << "..." << std::endl;
                
                LogisticRegression model(lr, 2000, 50, lambda, use_momentum, 0.9, use_all_features);
                model.fit(train_data, val_data, false);
                double val_acc = model.calculate_accuracy(val_data);
                
                std::cout << "  Validation Accuracy: " << val_acc << "%" << std::endl;
                
                if (val_acc > best_val_acc) {
                    best_val_acc = val_acc;
                    best_lr = lr;
                    best_lambda = lambda;
                    best_momentum = use_momentum;
                }
            }
        }
    }
    
    std::cout << "\nBest hyperparameters found:" << std::endl;
    std::cout << "  Learning Rate: " << best_lr << std::endl;
    std::cout << "  L2 Lambda: " << best_lambda << std::endl;
    std::cout << "  Momentum: " << (best_momentum ? "yes" : "no") << std::endl;
    std::cout << "  Validation Accuracy: " << best_val_acc << "%" << std::endl;
    
    // Create and return model with best hyperparameters
    return LogisticRegression(best_lr, 10000, 500, best_lambda, best_momentum, 0.9, use_all_features);
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
    std::vector<double> min_values, max_values;
    
    // 데이터셋 정규화
    normalize_features(dataset, min_values, max_values);
    
    // Split data into training and validation sets
    auto [train_data, val_data] = train_test_split(dataset, 0.25);
    
    std::cout << "Training with " << train_data.size() << " examples" << std::endl;
    std::cout << "Validating with " << val_data.size() << " examples" << std::endl;
    
    // Ask user if they want to run hyperparameter tuning
    char run_tuning;
    std::cout << "\nDo you want to run hyperparameter tuning? (y/n): ";
    std::cin >> run_tuning;
    
    // Ask if user wants to use engineered features
    char use_engineered;
    std::cout << "Do you want to use engineered features (interactions and polynomial)? (y/n): ";
    std::cin >> use_engineered;
    bool use_all_features = (use_engineered == 'y' || use_engineered == 'Y');
    
    LogisticRegression model;
    
    if (run_tuning == 'y' || run_tuning == 'Y') {
        // Tune hyperparameters
        model = tune_hyperparameters(train_data, val_data, use_all_features);
    } else {
        // Use default hyperparameters
        model = LogisticRegression(0.01, 5000, 500, 0.01, true, 0.9, use_all_features);
    }
    
    // Train the model
    model.fit(train_data, val_data);
    
    // Run cross-validation
    char run_cv;
    std::cout << "\nDo you want to run cross-validation? (y/n): ";
    std::cin >> run_cv;
    
    if (run_cv == 'y' || run_cv == 'Y') {
        model.cross_validate(dataset, 5);
    }
    
    // Evaluate model on validation set
    Metrics metrics = model.evaluate(val_data);
    std::cout << "\nValidation Metrics:" << std::endl;
    std::cout << "  Accuracy: " << metrics.accuracy << "%" << std::endl;
    std::cout << "  Precision: " << metrics.precision << "%" << std::endl;
    std::cout << "  Recall: " << metrics.recall << "%" << std::endl;
    std::cout << "  F1 Score: " << metrics.f1_score << "%" << std::endl;
    std::cout << "  AUC: " << metrics.auc << "%" << std::endl;
    
    // Save model
    char save_model_choice;
    std::cout << "\nDo you want to save the model? (y/n): ";
    std::cin >> save_model_choice;
    
    if (save_model_choice == 'y' || save_model_choice == 'Y') {
        std::string filename;
        std::cout << "Enter filename: ";
        std::cin >> filename;
        model.save_model(filename);
    }
    
    // Prediction loop
    while (true) {
        // Ask user for prediction inputs
        std::cout << "\n=== Prediction Mode ===" << std::endl;
        std::cout << "Enter study hours and IQ (e.g., 5.0 120) or -1 to exit: ";
        double study_hours, iq;
        std::cin >> study_hours;
        
        if (study_hours < 0) break;
        
        std::cin >> iq;
        
        // Adjust IQ to be relative to baseline
        double adjusted_iq = iq - baseline_iq;
        
        // Create a data point and normalize it
        DataPoint new_point(study_hours, adjusted_iq, -1);  // -1 means label unknown
        normalize_datapoint(new_point, min_values, max_values);
        
        // Make prediction
        double prob = model.predict_probability(new_point);
        int predicted_class = model.predict_class(new_point);
        
        // Display results with interpretation
        std::cout << "\nPrediction Results:" << std::endl;
        std::cout << "Study Hours: " << study_hours << " (normalized: " << new_point.x1 << ")" << std::endl;
        std::cout << "IQ: " << iq << " (adjusted: " << adjusted_iq 
                  << ", normalized: " << new_point.x2 << ")" << std::endl;
        
        if (use_all_features) {
            std::cout << "Interaction (study*IQ): " << new_point.x3 << std::endl;
            std::cout << "Study^2: " << new_point.x4 << std::endl;
            std::cout << "IQ^2: " << new_point.x5 << std::endl;
        }
        
        std::cout << "Probability of passing: " << std::fixed << std::setprecision(2) 
                  << prob * 100 << "%" << std::endl;
        std::cout << "Predicted outcome: " << (predicted_class ? "PASS" : "FAIL") << std::endl;
    }
    
    return 0;
}