// decision_tree.cpp - Decision Tree Implementation in C++17
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <optional>
#include <random>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <utility>
#include <functional>
#include <type_traits>
#include <sstream>
#include <chrono>

namespace ml {

// Helper for checking if a type is numeric (replacement for C++20 concepts)
template<typename T>
struct is_numeric : std::is_arithmetic<T> {};

template<typename T>
constexpr bool is_numeric_v = is_numeric<T>::value;

// Decision Tree supporting both categorical and numerical features
template<typename T = double, typename = std::enable_if_t<is_numeric_v<T>>>
class DecisionTree {
public:
    // Criterion for splitting nodes
    enum class Criterion {
        Gini,       // Gini impurity (default for classification)
        Entropy,    // Information gain (alternative for classification)
        MSE         // Mean squared error (for regression)
    };

    // Struct to hold dataset with features and labels
    struct Dataset {
        std::vector<std::vector<T>> features;  // [n_samples, n_features]
        std::vector<T> labels;                 // [n_samples]
        std::vector<std::string> feature_names; // Optional feature names
        
        // Validate that the dataset dimensions match
        bool validate() const {
            if (features.empty() || labels.empty()) {
                return false;
            }
            size_t n_samples = features.size();
            if (labels.size() != n_samples) {
                return false;
            }
            // Check that all feature vectors have the same length
            size_t n_features = features[0].size();
            for (const auto& sample : features) {
                if (sample.size() != n_features) {
                    return false;
                }
            }
            // Check feature names if provided
            if (!feature_names.empty() && feature_names.size() != n_features) {
                return false;
            }
            return true;
        }
    };

private:
    // Node structure for the decision tree
    struct Node {
        bool is_leaf = false;
        T value = T{};                  // For leaf nodes: prediction value
        size_t feature_index = 0;       // For non-leaf nodes: feature to split on
        T threshold = T{};              // For non-leaf nodes: threshold for splitting
        double impurity = 0.0;          // Node impurity (gini/entropy/mse)
        size_t n_samples = 0;           // Number of samples in the node
        
        std::shared_ptr<Node> left;     // Left child node (≤ threshold)
        std::shared_ptr<Node> right;    // Right child node (> threshold)
        
        // Constructor for a leaf node
        Node(T prediction_value, double node_impurity, size_t samples)
            : is_leaf(true), value(prediction_value), impurity(node_impurity), n_samples(samples) {}
        
        // Constructor for a decision node
        Node(size_t feat_index, T split_threshold, double node_impurity, size_t samples)
            : is_leaf(false), feature_index(feat_index), threshold(split_threshold),
              impurity(node_impurity), n_samples(samples) {}
    };

public:
    // Configuration struct for tree parameters
    struct Config {
        Criterion criterion = Criterion::Gini;  // Splitting criterion
        size_t max_depth = 0;                   // Max depth (0 = unlimited)
        size_t min_samples_split = 2;           // Min samples required to split an internal node
        size_t min_samples_leaf = 1;            // Min samples required in a leaf node
        size_t max_features = 0;                // Max features to consider (0 = all)
        T min_impurity_decrease = T{0};         // Min impurity decrease required for splitting
        bool random_state = false;              // Whether to use random state
        unsigned int seed = 42;                 // Random seed
    };

    // Constructor with configuration
    explicit DecisionTree(Config config = Config())
        : config_(std::move(config)), 
          random_engine_(config.seed),
          root_(nullptr) {}

    // Train the decision tree with a dataset
    void fit(const Dataset& dataset) {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety for training

        // Validate input data
        if (!dataset.validate()) {
            throw std::invalid_argument("Invalid dataset: dimensions mismatch or empty data");
        }

        // Store feature count
        n_features_ = dataset.features[0].size();
        
        // Use feature names if provided, otherwise generate generic names
        if (!dataset.feature_names.empty()) {
            feature_names_ = dataset.feature_names;
        } else {
            feature_names_.resize(n_features_);
            for (size_t i = 0; i < n_features_; ++i) {
                // Using stringstream instead of std::format
                std::stringstream ss;
                ss << "feature_" << i;
                feature_names_[i] = ss.str();
            }
        }

        // Find unique labels and their counts
        classes_ = get_unique_values(dataset.labels);
        
        // Initialize indices for the root node (all samples)
        std::vector<size_t> indices(dataset.features.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Build the tree recursively
        root_ = build_tree(dataset, indices, 0);
    }

    // Predict a single sample
    T predict(const std::vector<T>& features) const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety for prediction
        
        if (!root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }
        
        if (features.size() != n_features_) {
            // Using stringstream instead of std::format
            std::stringstream ss;
            ss << "Feature size mismatch. Expected " << n_features_ 
               << ", got " << features.size();
            throw std::invalid_argument(ss.str());
        }
        
        return predict_sample(features, root_);
    }

    // Predict multiple samples
    std::vector<T> predict(const std::vector<std::vector<T>>& features) const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety for prediction
        
        if (!root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }
        
        std::vector<T> predictions;
        predictions.reserve(features.size());
        
        for (const auto& sample : features) {
            if (sample.size() != n_features_) {
                // Using stringstream instead of std::format
                std::stringstream ss;
                ss << "Feature size mismatch at sample. Expected " << n_features_ 
                   << ", got " << sample.size();
                throw std::invalid_argument(ss.str());
            }
            predictions.push_back(predict_sample(sample, root_));
        }
        
        return predictions;
    }

    // Get feature importance (higher values mean more important features)
    std::vector<double> feature_importance() const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety
        
        if (!root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }
        
        std::vector<double> importance(n_features_, 0.0);
        compute_feature_importance(root_, importance);
        
        // Normalize to sum to 1
        double total = std::accumulate(importance.begin(), importance.end(), 0.0);
        if (total > 0) {
            for (auto& value : importance) {
                value /= total;
            }
        }
        
        return importance;
    }

    // Get the maximum depth of the trained tree
    size_t get_depth() const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety
        
        if (!root_) {
            return 0;
        }
        
        return compute_depth(root_);
    }

    // Get number of nodes in the tree
    size_t get_node_count() const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety
        
        if (!root_) {
            return 0;
        }
        
        return count_nodes(root_);
    }

    // Print the tree structure (for debugging)
    void print_tree() const {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread safety
        
        if (!root_) {
            std::cout << "Tree not trained yet" << std::endl;
            return;
        }
        
        print_node(root_, 0);
    }

private:
    Config config_;                       // Tree configuration
    mutable std::mutex mutex_;            // Mutex for thread-safety
    mutable std::mt19937 random_engine_;  // Random number generator (mutable for const methods)
    size_t n_features_ = 0;               // Number of features
    std::vector<T> classes_;              // Unique classes
    std::vector<std::string> feature_names_; // Feature names
    std::shared_ptr<Node> root_;          // Root node of the tree

    // Get unique values from a vector
    static std::vector<T> get_unique_values(const std::vector<T>& values) {
        std::vector<T> unique = values;
        std::sort(unique.begin(), unique.end());
        auto last = std::unique(unique.begin(), unique.end());
        unique.erase(last, unique.end());
        return unique;
    }

    // Count the occurrences of each unique value
    static std::unordered_map<T, size_t> count_values(const std::vector<T>& values) {
        std::unordered_map<T, size_t> counts;
        for (const auto& val : values) {
            counts[val]++;
        }
        return counts;
    }

    // Count the occurrences of each unique value in a subset
    static std::unordered_map<T, size_t> count_values(
        const std::vector<T>& values,
        const std::vector<size_t>& indices) {
        
        std::unordered_map<T, size_t> counts;
        for (size_t idx : indices) {
            counts[values[idx]]++;
        }
        return counts;
    }

    // Calculate entropy: -sum(p_i * log(p_i))
    static double calculate_entropy(const std::unordered_map<T, size_t>& class_counts, size_t total) {
        double entropy = 0.0;
        for (const auto& pair : class_counts) {
            double p = static_cast<double>(pair.second) / total;
            if (p > 0) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    }

    // Calculate Gini impurity: 1 - sum(p_i^2)
    static double calculate_gini(const std::unordered_map<T, size_t>& class_counts, size_t total) {
        double gini = 1.0;
        for (const auto& pair : class_counts) {
            double p = static_cast<double>(pair.second) / total;
            gini -= p * p;
        }
        return gini;
    }

    // Calculate MSE (Mean Squared Error) for regression
    static double calculate_mse(const std::vector<T>& values, const std::vector<size_t>& indices) {
        if (indices.empty()) return 0.0;
        
        // Calculate mean
        T sum = 0;
        for (size_t idx : indices) {
            sum += values[idx];
        }
        T mean = sum / indices.size();
        
        // Calculate MSE
        double mse = 0.0;
        for (size_t idx : indices) {
            double diff = values[idx] - mean;
            mse += diff * diff;
        }
        return mse / indices.size();
    }
    
    // Find the most common value (or mean for regression)
    static T get_prediction_value(
        const std::vector<T>& values,
        const std::vector<size_t>& indices,
        Criterion criterion) {
        
        if (criterion == Criterion::MSE) {
            // For regression, use mean
            T sum = 0;
            for (size_t idx : indices) {
                sum += values[idx];
            }
            return sum / indices.size();
        } else {
            // For classification, use most common class
            std::unordered_map<T, size_t> counts;
            for (size_t idx : indices) {
                counts[values[idx]]++;
            }
            
            T most_common{};
            size_t max_count = 0;
            for (const auto& pair : counts) {
                if (pair.second > max_count) {
                    most_common = pair.first;
                    max_count = pair.second;
                }
            }
            return most_common;
        }
    }

    // Calculate node impurity based on the criterion
    double calculate_impurity(
        const std::vector<T>& values,
        const std::vector<size_t>& indices) const {
        
        if (indices.empty()) return 0.0;
        
        switch (config_.criterion) {
            case Criterion::Gini: {
                auto counts = count_values(values, indices);
                return calculate_gini(counts, indices.size());
            }
            case Criterion::Entropy: {
                auto counts = count_values(values, indices);
                return calculate_entropy(counts, indices.size());
            }
            case Criterion::MSE:
                return calculate_mse(values, indices);
            default:
                throw std::invalid_argument("Unknown criterion");
        }
    }

    // Find the best split for a node
    std::optional<std::tuple<size_t, T, double, std::vector<size_t>, std::vector<size_t>>>
    find_best_split(const Dataset& dataset, const std::vector<size_t>& indices) const {
        if (indices.size() < config_.min_samples_split) {
            return std::nullopt;
        }
        
        double best_gain = 0.0;
        size_t best_feature = 0;
        T best_threshold = T{};
        std::vector<size_t> best_left_indices;
        std::vector<size_t> best_right_indices;
        
        double current_impurity = calculate_impurity(dataset.labels, indices);
        
        // Determine which features to consider
        std::vector<size_t> feature_indices(n_features_);
        std::iota(feature_indices.begin(), feature_indices.end(), 0);
        
        // Random feature selection if max_features specified
        if (config_.max_features > 0 && config_.max_features < n_features_) {
            // Create a copy of the random engine to use in this const method
            std::mt19937 temp_engine = random_engine_;
            std::shuffle(feature_indices.begin(), feature_indices.end(), temp_engine);
            feature_indices.resize(config_.max_features);
        }
        
        // Try splitting on each feature
        for (size_t feature_idx : feature_indices) {
            // Get unique values for this feature
            std::vector<T> feature_values;
            feature_values.reserve(indices.size());
            
            for (size_t idx : indices) {
                feature_values.push_back(dataset.features[idx][feature_idx]);
            }
            
            std::sort(feature_values.begin(), feature_values.end());
            auto last = std::unique(feature_values.begin(), feature_values.end());
            feature_values.erase(last, feature_values.end());
            
            // Skip features with only one unique value
            if (feature_values.size() < 2) {
                continue;
            }
            
            // Try different thresholds for this feature
            for (size_t i = 0; i < feature_values.size() - 1; ++i) {
                T threshold = (feature_values[i] + feature_values[i + 1]) / 2;
                
                std::vector<size_t> left_indices;
                std::vector<size_t> right_indices;
                
                // Split based on threshold
                for (size_t idx : indices) {
                    if (dataset.features[idx][feature_idx] <= threshold) {
                        left_indices.push_back(idx);
                    } else {
                        right_indices.push_back(idx);
                    }
                }
                
                // Skip if min_samples_leaf constraint is not satisfied
                if (left_indices.size() < config_.min_samples_leaf || 
                    right_indices.size() < config_.min_samples_leaf) {
                    continue;
                }
                
                // Calculate impurity for children
                double left_impurity = calculate_impurity(dataset.labels, left_indices);
                double right_impurity = calculate_impurity(dataset.labels, right_indices);
                
                // Calculate weighted impurity
                double n = static_cast<double>(indices.size());
                double weighted_impurity = (left_indices.size() / n) * left_impurity +
                                          (right_indices.size() / n) * right_impurity;
                
                // Calculate information gain
                double gain = current_impurity - weighted_impurity;
                
                // Update best split if this one is better
                if (gain > best_gain && gain >= config_.min_impurity_decrease) {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                    best_left_indices = std::move(left_indices);
                    best_right_indices = std::move(right_indices);
                }
            }
        }
        
        if (best_gain > 0) {
            return std::make_tuple(
                best_feature, best_threshold, best_gain,
                std::move(best_left_indices), std::move(best_right_indices));
        }
        
        return std::nullopt;  // No good split found
    }

    // Build the decision tree recursively
    std::shared_ptr<Node> build_tree(
        const Dataset& dataset,
        const std::vector<size_t>& indices,
        size_t depth) {
        
        // Check stopping criteria
        if (indices.empty()) {
            // Should not happen, but handle anyway
            return nullptr;
        }
        
        // Calculate current impurity
        double current_impurity = calculate_impurity(dataset.labels, indices);
        
        // Get the prediction value for this node
        T node_value = get_prediction_value(dataset.labels, indices, config_.criterion);
        
        // Check if we've reached max depth or have too few samples to split
        if ((config_.max_depth > 0 && depth >= config_.max_depth) || 
            indices.size() < config_.min_samples_split) {
            return std::make_shared<Node>(node_value, current_impurity, indices.size());
        }
        
        // Find the best split
        auto split_result = find_best_split(dataset, indices);
        
        // Create a leaf node if no good split is found
        if (!split_result) {
            return std::make_shared<Node>(node_value, current_impurity, indices.size());
        }
        
        // C++17 structured binding
        auto [feature_idx, threshold, gain, left_indices, right_indices] = *split_result;
        
        // Create a decision node
        auto node = std::make_shared<Node>(feature_idx, threshold, current_impurity, indices.size());
        
        // Recursively build left and right subtrees
        node->left = build_tree(dataset, left_indices, depth + 1);
        node->right = build_tree(dataset, right_indices, depth + 1);
        
        return node;
    }

    // Predict a single sample using the trained tree
    T predict_sample(const std::vector<T>& features, std::shared_ptr<Node> node) const {
        if (!node) {
            throw std::runtime_error("Invalid tree node");
        }
        
        if (node->is_leaf) {
            return node->value;
        }
        
        if (features[node->feature_index] <= node->threshold) {
            return predict_sample(features, node->left);
        } else {
            return predict_sample(features, node->right);
        }
    }

    // Compute feature importance recursively
    void compute_feature_importance(
        const std::shared_ptr<Node>& node,
        std::vector<double>& importance) const {
        
        if (!node || node->is_leaf) {
            return;
        }
        
        // Non-leaf nodes contribute to feature importance
        // Node importance is proportional to the number of samples it impacted
        // and the impurity decrease it achieved
        double node_importance = node->n_samples * node->impurity;
        
        if (node->left) {
            double left_importance = node->left->n_samples * node->left->impurity;
            node_importance -= left_importance;
        }
        
        if (node->right) {
            double right_importance = node->right->n_samples * node->right->impurity;
            node_importance -= right_importance;
        }
        
        importance[node->feature_index] += node_importance;
        
        // Recursive call for child nodes
        compute_feature_importance(node->left, importance);
        compute_feature_importance(node->right, importance);
    }

    // Compute the depth of the tree recursively
    static size_t compute_depth(const std::shared_ptr<Node>& node) {
        if (!node) {
            return 0;
        }
        
        if (node->is_leaf) {
            return 1;
        }
        
        return 1 + std::max(compute_depth(node->left), compute_depth(node->right));
    }

    // Count the number of nodes in the tree recursively
    static size_t count_nodes(const std::shared_ptr<Node>& node) {
        if (!node) {
            return 0;
        }
        
        return 1 + count_nodes(node->left) + count_nodes(node->right);
    }

    // Print the tree structure (for debugging)
    void print_node(const std::shared_ptr<Node>& node, size_t depth) const {
        if (!node) {
            return;
        }
        
        std::string indent(depth * 4, ' ');
        
        if (node->is_leaf) {
            std::cout << indent << "Leaf: value=" << node->value
                      << ", samples=" << node->n_samples
                      << ", impurity=" << node->impurity << std::endl;
        } else {
            std::cout << indent << "Node: feature=" << feature_names_[node->feature_index]
                      << ", threshold=" << node->threshold
                      << ", samples=" << node->n_samples
                      << ", impurity=" << node->impurity << std::endl;
            
            print_node(node->left, depth + 1);
            print_node(node->right, depth + 1);
        }
    }
};

} // namespace ml

// Helper function to print a confusion matrix for classification results
template<typename T>
void print_confusion_matrix(const std::vector<T>& true_labels, const std::vector<T>& predictions) {
    // Get unique labels
    std::vector<T> unique_labels = true_labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    auto last = std::unique(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(last, unique_labels.end());
    
    // Build confusion matrix
    std::cout << "\nConfusion Matrix:\n";
    std::cout << std::setw(10) << "Actual\\Pred";
    
    for (const auto& label : unique_labels) {
        std::cout << std::setw(10) << label;
    }
    std::cout << "\n";
    
    for (const auto& true_label : unique_labels) {
        std::cout << std::setw(10) << true_label;
        
        for (const auto& pred_label : unique_labels) {
            size_t count = 0;
            for (size_t i = 0; i < true_labels.size(); ++i) {
                if (true_labels[i] == true_label && predictions[i] == pred_label) {
                    count++;
                }
            }
            std::cout << std::setw(10) << count;
        }
        std::cout << "\n";
    }
}

// Function to calculate accuracy for classification
template<typename T>
double calculate_accuracy(const std::vector<T>& true_labels, const std::vector<T>& predictions) {
    if (true_labels.size() != predictions.size() || true_labels.empty()) {
        return 0.0;
    }
    
    size_t correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predictions[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / true_labels.size();
}

// Generate a simple dataset for testing
ml::DecisionTree<double>::Dataset generate_simple_dataset() {
    // Create a simple dataset with 2 features
    // Feature 1: x-coordinate, Feature 2: y-coordinate
    // Label: 0 if x + y <= 1, 1 otherwise
    
    const size_t n_samples = 1000;
    std::vector<std::vector<double>> features(n_samples, std::vector<double>(2));
    std::vector<double> labels(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2.0);
    
    for (size_t i = 0; i < n_samples; ++i) {
        features[i][0] = dis(gen);  // x
        features[i][1] = dis(gen);  // y
        
        // Decision boundary: x + y <= 1
        labels[i] = (features[i][0] + features[i][1] <= 1.0) ? 0.0 : 1.0;
    }
    
    return {features, labels, {"x", "y"}};
}

// Generate the Iris dataset (simplified version for testing)
ml::DecisionTree<double>::Dataset generate_iris_dataset() {
    // Simplified iris dataset with 3 features and 3 classes
    // Features: sepal length, sepal width, petal length
    // Labels: 0 (setosa), 1 (versicolor), 2 (virginica)
    
    const size_t n_samples = 150;  // 50 samples for each class
    std::vector<std::vector<double>> features(n_samples, std::vector<double>(3));
    std::vector<double> labels(n_samples);
    
    // Class 0: Setosa (well separated from others)
    for (size_t i = 0; i < 50; ++i) {
        // Setosa has small petals and wide sepals
        features[i][0] = 5.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal length
        features[i][1] = 3.5 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal width
        features[i][2] = 1.5 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // petal length
        labels[i] = 0.0;  // Setosa
    }
    
    // Class 1: Versicolor (somewhat overlaps with virginica)
    for (size_t i = 50; i < 100; ++i) {
        features[i][0] = 6.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal length
        features[i][1] = 2.5 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal width
        features[i][2] = 4.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // petal length
        labels[i] = 1.0;  // Versicolor
    }
    
    // Class 2: Virginica
    for (size_t i = 100; i < 150; ++i) {
        features[i][0] = 6.5 + 0.6 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal length
        features[i][1] = 3.0 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // sepal width
        features[i][2] = 5.5 + 0.6 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // petal length
        labels[i] = 2.0;  // Virginica
    }
    
    return {features, labels, {"sepal_length", "sepal_width", "petal_length"}};
}

// Split dataset into training and testing sets
template<typename T>
std::pair<typename ml::DecisionTree<T>::Dataset, typename ml::DecisionTree<T>::Dataset> 
train_test_split(const typename ml::DecisionTree<T>::Dataset& dataset, double test_size = 0.2) {
    if (dataset.features.empty() || dataset.labels.empty()) {
        throw std::invalid_argument("Empty dataset");
    }
    
    size_t n_samples = dataset.features.size();
    size_t n_test = static_cast<size_t>(n_samples * test_size);
    size_t n_train = n_samples - n_test;
    
    // Create indices and shuffle them
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Create training and testing datasets
    typename ml::DecisionTree<T>::Dataset train_dataset;
    typename ml::DecisionTree<T>::Dataset test_dataset;
    
    train_dataset.features.resize(n_train);
    train_dataset.labels.resize(n_train);
    train_dataset.feature_names = dataset.feature_names;
    
    test_dataset.features.resize(n_test);
    test_dataset.labels.resize(n_test);
    test_dataset.feature_names = dataset.feature_names;
    
    for (size_t i = 0; i < n_train; ++i) {
        train_dataset.features[i] = dataset.features[indices[i]];
        train_dataset.labels[i] = dataset.labels[indices[i]];
    }
    
    for (size_t i = 0; i < n_test; ++i) {
        test_dataset.features[i] = dataset.features[indices[n_train + i]];
        test_dataset.labels[i] = dataset.labels[indices[n_train + i]];
    }
    
    return {train_dataset, test_dataset};
}

// Test case for the simple dataset
void test_simple_dataset() {
    std::cout << "\n===== Test with Simple Dataset =====\n";
    
    // Generate dataset
    auto dataset = generate_simple_dataset();
    std::cout << "Dataset generated: " << dataset.features.size() << " samples, " 
              << dataset.features[0].size() << " features\n";
    
    // Split into training and testing sets
    auto split_result = train_test_split<double>(dataset, 0.2);
    auto train_dataset = split_result.first;
    auto test_dataset = split_result.second;
    
    std::cout << "Training set: " << train_dataset.features.size() << " samples\n";
    std::cout << "Testing set: " << test_dataset.features.size() << " samples\n";
    
    // Create and configure decision tree
    ml::DecisionTree<double>::Config config;
    config.criterion = ml::DecisionTree<double>::Criterion::Gini;
    config.max_depth = 5;
    config.min_samples_split = 2;
    config.min_samples_leaf = 1;
    
    ml::DecisionTree<double> tree(config);
    
    // Train the tree
    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(train_dataset);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> training_time = end - start;
    std::cout << "Training time: " << training_time.count() << " ms\n";
    
    // Make predictions
    start = std::chrono::high_resolution_clock::now();
    std::vector<double> predictions = tree.predict(test_dataset.features);
    end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> prediction_time = end - start;
    std::cout << "Prediction time: " << prediction_time.count() << " ms\n";
    
    // Calculate accuracy
    double accuracy = calculate_accuracy(test_dataset.labels, predictions);
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
    
    // Print confusion matrix
    print_confusion_matrix(test_dataset.labels, predictions);
    
    // Print tree statistics
    std::cout << "Tree depth: " << tree.get_depth() << "\n";
    std::cout << "Number of nodes: " << tree.get_node_count() << "\n";
    
    // Print feature importance
    std::vector<double> importance = tree.feature_importance();
    std::cout << "Feature importance:\n";
    for (size_t i = 0; i < importance.size(); ++i) {
        std::cout << "  " << dataset.feature_names[i] << ": " 
                  << std::fixed << std::setprecision(4) << importance[i] << "\n";
    }
    
    // Print tree structure
    std::cout << "\nTree structure:\n";
    tree.print_tree();
}

// Test case for the Iris dataset
void test_iris_dataset() {
    std::cout << "\n===== Test with Iris Dataset =====\n";
    
    // Generate dataset
    auto dataset = generate_iris_dataset();
    std::cout << "Dataset generated: " << dataset.features.size() << " samples, " 
              << dataset.features[0].size() << " features\n";
    
    // Split into training and testing sets
    auto split_result = train_test_split<double>(dataset, 0.2);
    auto train_dataset = split_result.first;
    auto test_dataset = split_result.second;
    
    std::cout << "Training set: " << train_dataset.features.size() << " samples\n";
    std::cout << "Testing set: " << test_dataset.features.size() << " samples\n";
    
    // Create and configure decision tree
    ml::DecisionTree<double>::Config config;
    config.criterion = ml::DecisionTree<double>::Criterion::Entropy;  // Using entropy for Iris
    config.max_depth = 4;
    
    ml::DecisionTree<double> tree(config);
    
    // Train the tree
    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(train_dataset);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> training_time = end - start;
    std::cout << "Training time: " << training_time.count() << " ms\n";
    
    // Make predictions
    start = std::chrono::high_resolution_clock::now();
    std::vector<double> predictions = tree.predict(test_dataset.features);
    end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> prediction_time = end - start;
    std::cout << "Prediction time: " << prediction_time.count() << " ms\n";
    
    // Calculate accuracy
    double accuracy = calculate_accuracy(test_dataset.labels, predictions);
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
    
    // Print confusion matrix
    print_confusion_matrix(test_dataset.labels, predictions);
    
    // Print feature importance
    std::vector<double> importance = tree.feature_importance();
    std::cout << "Feature importance:\n";
    for (size_t i = 0; i < importance.size(); ++i) {
        std::cout << "  " << dataset.feature_names[i] << ": " 
                  << std::fixed << std::setprecision(4) << importance[i] << "\n";
    }
}

// Test case for edge cases and error handling
void test_edge_cases() {
    std::cout << "\n===== Test Edge Cases and Error Handling =====\n";
    
    ml::DecisionTree<double> tree;
    
    // Test with empty dataset
    ml::DecisionTree<double>::Dataset empty_dataset;
    empty_dataset.features = {};
    empty_dataset.labels = {};
    
    try {
        tree.fit(empty_dataset);
        std::cout << "FAILED: Should have thrown exception for empty dataset\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "PASSED: Correctly caught exception for empty dataset: " << e.what() << "\n";
    }
    
    // Test with mismatched dimensions
    ml::DecisionTree<double>::Dataset mismatched_dataset;
    mismatched_dataset.features = {{1.0, 2.0}, {3.0, 4.0}};
    mismatched_dataset.labels = {1.0};  // Only one label for two samples
    
    try {
        tree.fit(mismatched_dataset);
        std::cout << "FAILED: Should have thrown exception for mismatched dimensions\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "PASSED: Correctly caught exception for mismatched dimensions: " << e.what() << "\n";
    }
    
    // Test prediction before training
    try {
        tree.predict(std::vector<double>{1.0, 2.0});  // Make the vector type explicit
        std::cout << "FAILED: Should have thrown exception for prediction before training\n";
    } catch (const std::runtime_error& e) {
        std::cout << "PASSED: Correctly caught exception for prediction before training: " << e.what() << "\n";
    }
    
    // Generate a valid dataset and train on it
    auto dataset = generate_simple_dataset();
    tree.fit(dataset);
    
    // Test prediction with wrong feature count
    try {
        tree.predict(std::vector<double>{1.0});  // Be explicit about the vector type
        std::cout << "FAILED: Should have thrown exception for wrong feature count\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "PASSED: Correctly caught exception for wrong feature count: " << e.what() << "\n";
    }
}

// Main function with performance benchmarks
int main() {
    std::cout << "Decision Tree Implementation in C++17\n";
    std::cout << "====================================\n";
    
    // Set random seed for reproducibility
    std::srand(42);
    
    // Run tests
    test_simple_dataset();
    test_iris_dataset();
    test_edge_cases();
    
    // Performance benchmark with larger dataset
    std::cout << "\n===== Performance Benchmark =====\n";
    
    // Create larger dataset (10000 samples, 5 features)
    const size_t n_samples = 10000;
    const size_t n_features = 5;
    
    std::vector<std::vector<double>> features(n_samples, std::vector<double>(n_features));
    std::vector<double> labels(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            features[i][j] = dis(gen);
        }
        
        // Decision boundary: sum of features > 25
        double sum = 0.0;
        for (size_t j = 0; j < n_features; ++j) {
            sum += features[i][j];
        }
        labels[i] = (sum > 25.0) ? 1.0 : 0.0;
    }
    
    ml::DecisionTree<double>::Dataset large_dataset = {
        features, labels, {"f1", "f2", "f3", "f4", "f5"}
    };
    
    // Split into training and testing sets
    auto split_result = train_test_split<double>(large_dataset, 0.2);
    auto train_dataset = split_result.first;
    auto test_dataset = split_result.second;
    
    // Benchmark with different max depths
    std::vector<size_t> depths = {5, 10, 15, 20};
    
    for (size_t depth : depths) {
        ml::DecisionTree<double>::Config config;
        config.max_depth = depth;
        
        ml::DecisionTree<double> tree(config);
        
        std::cout << "\nBenchmark with max_depth = " << depth << ":\n";
        
        // Measure training time
        auto start = std::chrono::high_resolution_clock::now();
        tree.fit(train_dataset);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> training_time = end - start;
        std::cout << "Training time: " << training_time.count() << " ms\n";
        
        // Measure prediction time
        start = std::chrono::high_resolution_clock::now();
        std::vector<double> predictions = tree.predict(test_dataset.features);
        end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> prediction_time = end - start;
        std::cout << "Prediction time: " << prediction_time.count() << " ms\n";
        
        // Calculate accuracy
        double accuracy = calculate_accuracy(test_dataset.labels, predictions);
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
        
        // Print tree statistics
        std::cout << "Tree depth: " << tree.get_depth() << "\n";
        std::cout << "Number of nodes: " << tree.get_node_count() << "\n";
    }
    
    return 0;
}