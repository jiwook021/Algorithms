#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <iomanip>
#include <limits>

/**
 * @brief Represents a data point with features and a class label
 * 
 * This class encapsulates a data point with two features (x1, x2) and
 * a binary class label (0 or 1).
 */
class DataPoint {
public:
    /**
     * @brief Construct a data point with features and an optional label
     * 
     * @param x1 First feature value
     * @param x2 Second feature value
     * @param label Optional class label (0 or 1, or nullopt if unknown)
     */
    DataPoint(double x1, double x2, std::optional<int> label = std::nullopt)
        : x1_(x1), x2_(x2), label_(label) {}

    // Getters
    double getX1() const { return x1_; }
    double getX2() const { return x2_; }
    std::optional<int> getLabel() const { return label_; }
    
    // Setters
    void setLabel(int label) { 
        if (label != 0 && label != 1) {
            throw std::invalid_argument("Label must be 0 or 1");
        }
        label_ = label; 
    }

    /**
     * @brief Create DataPoint from user input
     * 
     * @return DataPoint created from console input
     * @throw std::runtime_error If input is invalid
     */
    static DataPoint fromUserInput() {
        double x1, x2;
        std::cout << "Enter x1 and x2 features (e.g., 3.5 4.5): ";
        if (!(std::cin >> x1 >> x2)) {
            std::cin.clear();  // Clear error state
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard invalid input
            throw std::runtime_error("Invalid input format. Please enter two numeric values.");
        }
        return DataPoint(x1, x2);
    }

private:
    double x1_;  // First feature
    double x2_;  // Second feature
    std::optional<int> label_;  // Class label (optional)
};

/**
 * @brief Statistical distribution interface
 * 
 * Abstract base class for probability distributions used in the classifier
 */
class Distribution {
public:
    virtual ~Distribution() = default;
    
    /**
     * @brief Calculate probability density for a given value
     * 
     * @param x Value to calculate probability for
     * @return double Probability density
     */
    virtual double probabilityDensity(double x) const = 0;
    
    /**
     * @brief Calculate log probability density for a given value
     * 
     * @param x Value to calculate log probability for
     * @return double Log probability density
     */
    virtual double logProbabilityDensity(double x) const = 0;
    
    /**
     * @brief Clone the distribution
     * 
     * @return std::unique_ptr<Distribution> Cloned distribution
     */
    virtual std::unique_ptr<Distribution> clone() const = 0;
};

/**
 * @brief Gaussian (Normal) probability distribution
 * 
 * Implements a Gaussian probability distribution with mean and variance parameters
 */
class GaussianDistribution : public Distribution {
public:
    /**
     * @brief Construct a Gaussian distribution
     * 
     * @param mean Mean of the distribution
     * @param variance Variance of the distribution (must be positive)
     * @throw std::invalid_argument If variance is not positive
     */
    GaussianDistribution(double mean, double variance)
        : mean_(mean), variance_(variance) {
        if (variance <= 0.0) {
            throw std::invalid_argument("Variance must be positive");
        }
        stdDev_ = std::sqrt(variance);
        // Pre-calculate normalization factor to avoid redundant calculations
        constexpr double PI = 3.141592653589793;
        normalizationFactor_ = 1.0 / (stdDev_ * std::sqrt(2.0 * PI));
    }

    /**
     * @brief Calculate Gaussian probability density
     * 
     * @param x Value to calculate probability for
     * @return double Probability density
     */
    double probabilityDensity(double x) const override {
        double exponent = -std::pow(x - mean_, 2) / (2.0 * variance_);
        // Guard against extreme values to prevent underflow
        if (exponent < -700.0) return 0.0;
        return normalizationFactor_ * std::exp(exponent);
    }

    /**
     * @brief Calculate log of Gaussian probability density
     * 
     * @param x Value to calculate log probability for
     * @return double Log probability density
     */
    double logProbabilityDensity(double x) const override {
        double exponent = -std::pow(x - mean_, 2) / (2.0 * variance_);
        return std::log(normalizationFactor_) + exponent;
    }

    /**
     * @brief Clone this distribution
     * 
     * @return std::unique_ptr<Distribution> Cloned distribution
     */
    std::unique_ptr<Distribution> clone() const override {
        return std::make_unique<GaussianDistribution>(mean_, variance_);
    }

    // Getters
    double getMean() const { return mean_; }
    double getVariance() const { return variance_; }

private:
    double mean_;               // Mean of the distribution
    double variance_;           // Variance of the distribution
    double stdDev_;             // Standard deviation (cached)
    double normalizationFactor_;  // Pre-calculated normalization factor
};

/**
 * @brief Class for feature statistics used in Naive Bayes
 * 
 * This class encapsulates statistics for a single feature across samples of the same class
 */
class FeatureStats {
public:
    /**
     * @brief Default constructor
     */
    FeatureStats() : mean_(0.0), variance_(0.0), count_(0) {}

    /**
     * @brief Update statistics with a new value
     * 
     * Updates mean and variance using Welford's online algorithm for numerical stability
     * 
     * @param value New feature value to incorporate
     */
    void update(double value) {
        count_++;
        
        // For the first value, just set it as mean
        if (count_ == 1) {
            mean_ = value;
            m2_ = 0.0;
            return;
        }
        
        // Update using Welford's online algorithm for better numerical stability
        double delta = value - mean_;
        mean_ += delta / count_;
        double delta2 = value - mean_;
        m2_ += delta * delta2;
    }

    /**
     * @brief Finalize statistics calculation
     * 
     * @param minVariance Minimum allowed variance to prevent numerical issues
     */
    void finalize(double minVariance = 1e-6) {
        // Calculate variance, ensure it's at least minVariance
        variance_ = (count_ > 1) ? (m2_ / (count_ - 1)) : minVariance;
        variance_ = std::max(variance_, minVariance);
    }

    /**
     * @brief Create a distribution for this feature
     * 
     * @return std::unique_ptr<Distribution> Feature distribution
     */
    std::unique_ptr<Distribution> createDistribution() const {
        return std::make_unique<GaussianDistribution>(mean_, variance_);
    }

    // Getters
    double getMean() const { return mean_; }
    double getVariance() const { return variance_; }
    size_t getCount() const { return count_; }

private:
    double mean_;      // Mean of the feature
    double variance_;  // Variance of the feature
    double m2_;        // Sum of squared differences (for Welford's algorithm)
    size_t count_;     // Number of samples
};

/**
 * @brief Class for a dataset of data points
 * 
 * This class manages a collection of DataPoint objects and provides
 * operations for data manipulation and analysis.
 */
class DataSet {
public:
    /**
     * @brief Add a data point to the dataset
     * 
     * @param dataPoint The data point to add
     * @param requireLabels If true, data point must have a label
     * @throw std::invalid_argument If the data point has no label and requireLabels is true
     */
    void addDataPoint(const DataPoint& dataPoint, bool requireLabels = true) {
        if (requireLabels && !dataPoint.getLabel().has_value()) {
            throw std::invalid_argument("Data point must have a label");
        }
        dataPoints_.push_back(dataPoint);
    }

    /**
     * @brief Add multiple data points to the dataset
     * 
     * @param dataPoints Vector of data points to add
     * @param requireLabels If true, all data points must have labels
     * @throw std::invalid_argument If any data point has no label and requireLabels is true
     */
    void addDataPoints(const std::vector<DataPoint>& dataPoints, bool requireLabels = true) {
        for (const auto& dp : dataPoints) {
            addDataPoint(dp, requireLabels);
        }
    }

    /**
     * @brief Get all data points
     * 
     * @return const std::vector<DataPoint>& Reference to data points vector
     */
    const std::vector<DataPoint>& getDataPoints() const {
        return dataPoints_;
    }

    /**
     * @brief Get data points with a specific label
     * 
     * @param label The label to filter by
     * @return std::vector<DataPoint> Data points with the specified label
     */
    std::vector<DataPoint> getDataPointsByLabel(int label) const {
        std::vector<DataPoint> result;
        for (const auto& dp : dataPoints_) {
            auto dpLabel = dp.getLabel();
            if (dpLabel.has_value() && dpLabel.value() == label) {
                result.push_back(dp);
            }
        }
        return result;
    }

    /**
     * @brief Check if the dataset is empty
     * 
     * @return true If the dataset contains no data points
     * @return false If the dataset contains data points
     */
    bool isEmpty() const {
        return dataPoints_.empty();
    }

    /**
     * @brief Get the number of data points
     * 
     * @return size_t Number of data points
     */
    size_t size() const {
        return dataPoints_.size();
    }

    /**
     * @brief Check if the dataset is valid for training
     * 
     * @return bool True if dataset is valid, false otherwise
     */
    bool isValidForTraining() const {
        if (isEmpty()) {
            return false;
        }

        bool hasClass0 = false;
        bool hasClass1 = false;

        for (const auto& dp : dataPoints_) {
            auto label = dp.getLabel();
            if (!label.has_value()) {
                return false;  // All points must have labels
            }
            
            if (label.value() == 0) hasClass0 = true;
            if (label.value() == 1) hasClass1 = true;
            
            if (hasClass0 && hasClass1) break;  // Found both classes
        }

        return hasClass0 && hasClass1;  // Need at least one sample from each class
    }

    /**
     * @brief Create a sample dataset for testing
     * 
     * @return DataSet A sample dataset
     */
    static DataSet createSampleDataset() {
        DataSet dataset;
        
        // Class 0 samples
        dataset.addDataPoint(DataPoint(2.0, 3.0, 0));
        dataset.addDataPoint(DataPoint(1.0, 2.0, 0));
        dataset.addDataPoint(DataPoint(3.0, 4.0, 0));
        
        // Class 1 samples
        dataset.addDataPoint(DataPoint(5.0, 6.0, 1));
        dataset.addDataPoint(DataPoint(4.0, 5.0, 1));
        dataset.addDataPoint(DataPoint(6.0, 7.0, 1));
        
        return dataset;
    }

private:
    std::vector<DataPoint> dataPoints_;  // Collection of data points
};

/**
 * @brief Class for a trained Naive Bayes model
 * 
 * This class encapsulates the model parameters for a trained Naive Bayes classifier
 */
class NaiveBayesModel {
public:
    /**
     * @brief Construct an empty model
     */
    NaiveBayesModel() : class0Prior_(0.5), class1Prior_(0.5) {}

    /**
     * @brief Construct a model with priors and feature distributions
     * 
     * @param class0Prior Prior probability for class 0
     * @param class1Prior Prior probability for class 1
     * @param class0Distributions Feature distributions for class 0
     * @param class1Distributions Feature distributions for class 1
     */
    NaiveBayesModel(
        double class0Prior,
        double class1Prior,
        std::vector<std::unique_ptr<Distribution>> class0Distributions,
        std::vector<std::unique_ptr<Distribution>> class1Distributions
    ) : class0Prior_(class0Prior),
        class1Prior_(class1Prior),
        class0Distributions_(std::move(class0Distributions)),
        class1Distributions_(std::move(class1Distributions)) {}

    /**
     * @brief Copy constructor
     * 
     * @param other Model to copy
     */
    NaiveBayesModel(const NaiveBayesModel& other)
        : class0Prior_(other.class0Prior_),
          class1Prior_(other.class1Prior_) {
        // Deep copy distributions
        for (const auto& dist : other.class0Distributions_) {
            class0Distributions_.push_back(dist->clone());
        }
        for (const auto& dist : other.class1Distributions_) {
            class1Distributions_.push_back(dist->clone());
        }
    }

    /**
     * @brief Move constructor
     * 
     * @param other Model to move from
     */
    NaiveBayesModel(NaiveBayesModel&& other) noexcept = default;

    /**
     * @brief Copy assignment operator
     * 
     * @param other Model to copy
     * @return NaiveBayesModel& Reference to this model
     */
    NaiveBayesModel& operator=(const NaiveBayesModel& other) {
        if (this != &other) {
            class0Prior_ = other.class0Prior_;
            class1Prior_ = other.class1Prior_;
            
            // Clear and deep copy distributions
            class0Distributions_.clear();
            class1Distributions_.clear();
            
            for (const auto& dist : other.class0Distributions_) {
                class0Distributions_.push_back(dist->clone());
            }
            for (const auto& dist : other.class1Distributions_) {
                class1Distributions_.push_back(dist->clone());
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     * 
     * @param other Model to move from
     * @return NaiveBayesModel& Reference to this model
     */
    NaiveBayesModel& operator=(NaiveBayesModel&& other) noexcept = default;

    /**
     * @brief Calculate log probability for a class
     * 
     * @param features Vector of feature values
     * @param classDistributions Vector of feature distributions for the class
     * @param classPrior Prior probability for the class
     * @return double Log probability
     * @throw std::invalid_argument If features and distributions sizes don't match
     */
    double calculateLogProb(
        const std::vector<double>& features,
        const std::vector<std::unique_ptr<Distribution>>& classDistributions,
        double classPrior
    ) const {
        if (features.size() != classDistributions.size()) {
            throw std::invalid_argument("Number of features must match number of distributions");
        }
        
        double logProb = std::log(classPrior);
        
        for (size_t i = 0; i < features.size(); ++i) {
            logProb += classDistributions[i]->logProbabilityDensity(features[i]);
        }
        
        return logProb;
    }

    /**
     * @brief Predict class for a data point
     * 
     * @param dataPoint Data point to classify
     * @return int Predicted class (0 or 1)
     */
    int predict(const DataPoint& dataPoint) const {
        std::vector<double> features = {dataPoint.getX1(), dataPoint.getX2()};
        
        double logProb0 = calculateLogProb(features, class0Distributions_, class0Prior_);
        double logProb1 = calculateLogProb(features, class1Distributions_, class1Prior_);
        
        return (logProb1 > logProb0) ? 1 : 0;
    }

    /**
     * @brief Predict probabilities for a data point
     * 
     * @param dataPoint Data point to classify
     * @return std::pair<double, double> Probabilities for class 0 and class 1
     */
    std::pair<double, double> predictProba(const DataPoint& dataPoint) const {
        std::vector<double> features = {dataPoint.getX1(), dataPoint.getX2()};
        
        double logProb0 = calculateLogProb(features, class0Distributions_, class0Prior_);
        double logProb1 = calculateLogProb(features, class1Distributions_, class1Prior_);
        
        // Convert from log space to probability space (with numerical stability)
        double maxLogProb = std::max(logProb0, logProb1);
        double prob0 = std::exp(logProb0 - maxLogProb);
        double prob1 = std::exp(logProb1 - maxLogProb);
        
        // Normalize
        double sum = prob0 + prob1;
        return {prob0 / sum, prob1 / sum};
    }

    // Getters
    double getClass0Prior() const { return class0Prior_; }
    double getClass1Prior() const { return class1Prior_; }
    
    /**
     * @brief Get model summary
     * 
     * @return std::string Text summary of the model
     */
    std::string getSummary() const {
        std::string summary = "Naive Bayes Model Summary:\n";
        summary += "------------------------\n";
        
        // Format with precision
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        
        oss << "Class 0 Prior: " << class0Prior_ << "\n";
        oss << "Class 1 Prior: " << class1Prior_ << "\n\n";
        
        oss << "Feature 1 (X1):\n";
        if (!class0Distributions_.empty()) {
            auto* gaussDist0 = dynamic_cast<GaussianDistribution*>(class0Distributions_[0].get());
            if (gaussDist0) {
                oss << "  Class 0: Mean = " << gaussDist0->getMean() 
                    << ", Variance = " << gaussDist0->getVariance() << "\n";
            }
        }
        if (!class1Distributions_.empty()) {
            auto* gaussDist1 = dynamic_cast<GaussianDistribution*>(class1Distributions_[0].get());
            if (gaussDist1) {
                oss << "  Class 1: Mean = " << gaussDist1->getMean() 
                    << ", Variance = " << gaussDist1->getVariance() << "\n";
            }
        }
        
        oss << "\nFeature 2 (X2):\n";
        if (class0Distributions_.size() > 1) {
            auto* gaussDist0 = dynamic_cast<GaussianDistribution*>(class0Distributions_[1].get());
            if (gaussDist0) {
                oss << "  Class 0: Mean = " << gaussDist0->getMean() 
                    << ", Variance = " << gaussDist0->getVariance() << "\n";
            }
        }
        if (class1Distributions_.size() > 1) {
            auto* gaussDist1 = dynamic_cast<GaussianDistribution*>(class1Distributions_[1].get());
            if (gaussDist1) {
                oss << "  Class 1: Mean = " << gaussDist1->getMean() 
                    << ", Variance = " << gaussDist1->getVariance() << "\n";
            }
        }
        
        return oss.str();
    }

private:
    double class0Prior_;  // Prior probability for class 0
    double class1Prior_;  // Prior probability for class 1
    std::vector<std::unique_ptr<Distribution>> class0Distributions_;  // Feature distributions for class 0
    std::vector<std::unique_ptr<Distribution>> class1Distributions_;  // Feature distributions for class 1
};

/**
 * @brief Naive Bayes classifier for binary classification
 * 
 * This class implements a Naive Bayes classifier for binary classification
 * with continuous features. It assumes feature independence and Gaussian
 * distribution for each feature.
 * 
 * Time Complexity:
 *   - Training: O(n), where n is the number of data points
 *   - Prediction: O(f), where f is the number of features
 * 
 * Space Complexity: O(f), where f is the number of features
 */
class NaiveBayesClassifier {
public:
    /**
     * @brief Construct a new Naive Bayes classifier
     */
    NaiveBayesClassifier() : isTrained_(false) {}

    /**
     * @brief Train the classifier on a dataset
     * 
     * @param dataset Dataset to train on
     * @param minVariance Minimum variance to prevent numerical issues
     * @return true If training was successful
     * @return false If training failed
     * @throw std::runtime_error If the dataset is invalid
     */
    bool train(const DataSet& dataset, double minVariance = 1e-6) {
        std::lock_guard<std::mutex> lock(trainingMutex_);  // Protect against concurrent training
        
        if (!dataset.isValidForTraining()) {
            throw std::runtime_error("Invalid dataset for training");
        }
        
        // Get data points for each class
        auto dataPoints0 = dataset.getDataPointsByLabel(0);
        auto dataPoints1 = dataset.getDataPointsByLabel(1);
        
        if (dataPoints0.empty() || dataPoints1.empty()) {
            throw std::runtime_error("Each class must have at least one sample");
        }
        
        // Calculate priors
        double totalSamples = static_cast<double>(dataset.size());
        double class0Prior = static_cast<double>(dataPoints0.size()) / totalSamples;
        double class1Prior = static_cast<double>(dataPoints1.size()) / totalSamples;
        
        // Initialize feature statistics
        std::vector<FeatureStats> class0Stats(2);  // Two features
        std::vector<FeatureStats> class1Stats(2);
        
        // Collect statistics for class 0
        for (const auto& dp : dataPoints0) {
            class0Stats[0].update(dp.getX1());
            class0Stats[1].update(dp.getX2());
        }
        
        // Collect statistics for class 1
        for (const auto& dp : dataPoints1) {
            class1Stats[0].update(dp.getX1());
            class1Stats[1].update(dp.getX2());
        }
        
        // Finalize statistics
        for (auto& stats : class0Stats) stats.finalize(minVariance);
        for (auto& stats : class1Stats) stats.finalize(minVariance);
        
        // Create distributions
        std::vector<std::unique_ptr<Distribution>> class0Distributions;
        std::vector<std::unique_ptr<Distribution>> class1Distributions;
        
        for (const auto& stats : class0Stats) {
            class0Distributions.push_back(stats.createDistribution());
        }
        
        for (const auto& stats : class1Stats) {
            class1Distributions.push_back(stats.createDistribution());
        }
        
        // Create and store model
        model_ = NaiveBayesModel(
            class0Prior,
            class1Prior,
            std::move(class0Distributions),
            std::move(class1Distributions)
        );
        
        isTrained_ = true;
        return true;
    }

    /**
     * @brief Predict class for a data point
     * 
     * @param dataPoint Data point to classify
     * @return int Predicted class (0 or 1)
     * @throw std::runtime_error If the model is not trained
     */
    int predict(const DataPoint& dataPoint) const {
        if (!isTrained_) {
            throw std::runtime_error("Model not trained");
        }
        return model_.predict(dataPoint);
    }

    /**
     * @brief Predict probability for each class
     * 
     * @param dataPoint Data point to classify
     * @return std::pair<double, double> Probabilities for class 0 and 1
     * @throw std::runtime_error If the model is not trained
     */
    std::pair<double, double> predictProba(const DataPoint& dataPoint) const {
        if (!isTrained_) {
            throw std::runtime_error("Model not trained");
        }
        return model_.predictProba(dataPoint);
    }

    /**
     * @brief Check if the model is trained
     * 
     * @return true If the model is trained
     * @return false If the model is not trained
     */
    bool isTrained() const {
        return isTrained_;
    }

    /**
     * @brief Get model summary
     * 
     * @return std::string Text summary of the model
     * @throw std::runtime_error If the model is not trained
     */
    std::string getModelSummary() const {
        if (!isTrained_) {
            throw std::runtime_error("Model not trained");
        }
        return model_.getSummary();
    }

private:
    NaiveBayesModel model_;         // Trained model
    bool isTrained_;                // Flag indicating if the model is trained
    mutable std::mutex trainingMutex_;  // Mutex for thread-safe training (used to protect model state during training)
};

/**
 * @brief Run tests to verify the classifier
 * 
 * @return bool True if all tests pass, false otherwise
 */
bool runTests() {
    bool allTestsPassed = true;
    
    // Test 1: Create and train on sample dataset
    try {
        std::cout << "Test 1: Training on sample dataset... ";
        
        DataSet dataset = DataSet::createSampleDataset();
        NaiveBayesClassifier classifier;
        
        bool trained = classifier.train(dataset);
        
        if (!trained || !classifier.isTrained()) {
            std::cout << "FAILED (Training unsuccessful)" << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")" << std::endl;
        allTestsPassed = false;
    }
    
    // Test 2: Prediction on sample data
    try {
        std::cout << "Test 2: Prediction on sample data... ";
        
        DataSet dataset = DataSet::createSampleDataset();
        NaiveBayesClassifier classifier;
        classifier.train(dataset);
        
        // Create test points (should be class 0)
        DataPoint testPoint0(2.5, 3.5);
        int prediction0 = classifier.predict(testPoint0);
        
        // Create test points (should be class 1)
        DataPoint testPoint1(5.5, 6.5);
        int prediction1 = classifier.predict(testPoint1);
        
        if (prediction0 != 0 || prediction1 != 1) {
            std::cout << "FAILED (Incorrect predictions)" << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")" << std::endl;
        allTestsPassed = false;
    }
    
    // Test 3: Probability prediction
    try {
        std::cout << "Test 3: Probability prediction... ";
        
        DataSet dataset = DataSet::createSampleDataset();
        NaiveBayesClassifier classifier;
        classifier.train(dataset);
        
        // Create test point
        DataPoint testPoint(2.5, 3.5);  // Should be high probability for class 0
        auto [prob0, prob1] = classifier.predictProba(testPoint);
        
        if (prob0 <= 0.0 || prob0 >= 1.0 || prob1 <= 0.0 || prob1 >= 1.0 || std::abs(prob0 + prob1 - 1.0) > 1e-9) {
            std::cout << "FAILED (Invalid probabilities)" << std::endl;
            allTestsPassed = false;
        } else if (prob0 <= prob1) {  // Should have higher probability for class 0
            std::cout << "FAILED (Unexpected probability distribution)" << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")" << std::endl;
        allTestsPassed = false;
    }
    
    // Test 4: Empty dataset
    try {
        std::cout << "Test 4: Training on empty dataset... ";
        
        DataSet emptyDataset;
        NaiveBayesClassifier classifier;
        
        try {
            classifier.train(emptyDataset);
            std::cout << "FAILED (Should throw exception)" << std::endl;
            allTestsPassed = false;
        } catch (const std::runtime_error&) {
            std::cout << "PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (Unexpected exception: " << e.what() << ")" << std::endl;
        allTestsPassed = false;
    }
    
    // Test 5: Missing class
    try {
        std::cout << "Test 5: Training with missing class... ";
        
        DataSet datasetMissingClass;
        datasetMissingClass.addDataPoint(DataPoint(1.0, 2.0, 0));
        datasetMissingClass.addDataPoint(DataPoint(2.0, 3.0, 0));
        
        NaiveBayesClassifier classifier;
        
        try {
            classifier.train(datasetMissingClass);
            std::cout << "FAILED (Should throw exception)" << std::endl;
            allTestsPassed = false;
        } catch (const std::runtime_error&) {
            std::cout << "PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (Unexpected exception: " << e.what() << ")" << std::endl;
        allTestsPassed = false;
    }
    
    return allTestsPassed;
}

/**
 * @brief Main function demonstrating Naive Bayes classifier
 * 
 * @return int Exit code
 */
int main() {
    try {
        std::cout << "Naive Bayes Classifier for Binary Classification\n";
        std::cout << "==============================================\n\n";
        
        // Run tests if requested
        char runTestsOption;
        std::cout << "Run tests? (y/n): ";
        std::cin >> runTestsOption;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Clear input buffer
        
        if (runTestsOption == 'y' || runTestsOption == 'Y') {
            bool testsPassed = runTests();
            std::cout << "\nAll tests " << (testsPassed ? "PASSED" : "FAILED") << std::endl;
            
            if (!testsPassed) {
                std::cout << "Exiting due to test failures." << std::endl;
                return 1;
            }
            
            std::cout << std::endl;
        }
        
        // Create and train classifier
        DataSet dataset = DataSet::createSampleDataset();
        NaiveBayesClassifier classifier;
        
        std::cout << "Training classifier on sample dataset..." << std::endl;
        classifier.train(dataset);
        
        std::cout << "\nModel successfully trained.\n" << std::endl;
        std::cout << classifier.getModelSummary() << std::endl;
        
        // Prediction loop
        bool continuePredict = true;
        while (continuePredict) {
            try {
                // Get input from user
                DataPoint newDataPoint = DataPoint::fromUserInput();
                
                // Make prediction
                int prediction = classifier.predict(newDataPoint);
                auto [prob0, prob1] = classifier.predictProba(newDataPoint);
                
                // Show result
                std::cout << "Predicted class: " << prediction << std::endl;
                std::cout << "Prediction probabilities: Class 0: " << std::fixed << std::setprecision(4) 
                          << prob0 << ", Class 1: " << prob1 << std::endl;
                
                // Ask if user wants to continue
                char continueOption;
                std::cout << "\nMake another prediction? (y/n): ";
                std::cin >> continueOption;
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Clear input buffer
                
                continuePredict = (continueOption == 'y' || continueOption == 'Y');
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
        
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}