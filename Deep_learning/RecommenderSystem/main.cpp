#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <future>
#include <stdexcept>
#include <span>  // C++20 feature

// Forward declarations
class User;
class Item;
class Rating;
class DataLoader;
class ModelTrainer;
class Recommender;

/**
 * @class Rating
 * @brief Represents a user-item interaction with a rating value
 */
class Rating {
public:
    Rating(int userId, int itemId, float value) 
        : userId_(userId), itemId_(itemId), value_(value) {}
    
    int getUserId() const { return userId_; }
    int getItemId() const { return itemId_; }
    float getValue() const { return value_; }

private:
    int userId_;    // The ID of the user who provided the rating
    int itemId_;    // The ID of the item that was rated
    float value_;   // The rating value (typically on a scale, e.g., 1-5)
};

/**
 * @class DataLoader
 * @brief Responsible for loading and preprocessing rating data
 */
class DataLoader {
public:
    /**
     * @brief Loads ratings from a CSV file
     * 
     * @param filename Path to the CSV file containing ratings
     * @param hasHeader Whether the CSV file has a header row
     * @return std::vector<Rating> Vector of Rating objects
     * @throws std::runtime_error if the file cannot be opened or has invalid format
     */
    std::vector<Rating> loadRatingsFromCsv(const std::string& filename, bool hasHeader = true) {
        std::vector<Rating> ratings;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::string line;
        
        // Skip header if present
        if (hasHeader && std::getline(file, line)) {
            // Do nothing, just skipping the header
        }
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> cells;
            
            while (std::getline(ss, cell, ',')) {
                cells.push_back(cell);
            }
            
            if (cells.size() < 3) {
                throw std::runtime_error("Invalid CSV format: Each line must contain at least userId, itemId, and rating");
            }
            
            try {
                int userId = std::stoi(cells[0]);
                int itemId = std::stoi(cells[1]);
                float ratingValue = std::stof(cells[2]);
                
                ratings.emplace_back(userId, itemId, ratingValue);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing values from CSV: " + std::string(e.what()));
            }
        }
        
        return ratings;
    }
    
    /**
     * @brief Splits ratings into training and test sets
     * 
     * @param ratings The complete set of ratings
     * @param testFraction Fraction of ratings to use for testing (0-1)
     * @param seed Random seed for reproducibility
     * @return std::pair<std::vector<Rating>, std::vector<Rating>> Training and test sets
     */
    std::pair<std::vector<Rating>, std::vector<Rating>> splitTrainTest(
            const std::vector<Rating>& ratings, 
            float testFraction = 0.2,
            unsigned int seed = 42) {
        
        if (testFraction < 0.0 || testFraction > 1.0) {
            throw std::invalid_argument("testFraction must be between 0 and 1");
        }
        
        // Create a copy of the ratings that we can shuffle
        std::vector<Rating> ratingsCopy = ratings;
        
        // Shuffle the ratings
        std::mt19937 rng(seed);
        std::shuffle(ratingsCopy.begin(), ratingsCopy.end(), rng);
        
        // Calculate split point
        size_t testSize = static_cast<size_t>(ratingsCopy.size() * testFraction);
        size_t trainSize = ratingsCopy.size() - testSize;
        
        // Split the data
        std::vector<Rating> trainSet(ratingsCopy.begin(), ratingsCopy.begin() + trainSize);
        std::vector<Rating> testSet(ratingsCopy.begin() + trainSize, ratingsCopy.end());
        
        return {trainSet, testSet};
    }
};

/**
 * @class MatrixFactorizationModel
 * @brief Implements matrix factorization for collaborative filtering
 * 
 * This class implements a matrix factorization approach where user-item
 * ratings are approximated as a product of latent factor vectors.
 */
class MatrixFactorizationModel {
public:
    /**
     * @brief Constructs a matrix factorization model
     * 
     * @param numFactors Number of latent factors to use
     * @param learningRate Learning rate for gradient descent
     * @param regularization Regularization parameter to prevent overfitting
     * @param numIterations Number of iterations for training
     */
    MatrixFactorizationModel(
            int numFactors = 10,
            float learningRate = 0.01,
            float regularization = 0.02,
            int numIterations = 100)
        : numFactors_(numFactors),
          learningRate_(learningRate),
          regularization_(regularization),
          numIterations_(numIterations),
          initialized_(false) {}
    
    /**
     * @brief Initializes the model with user and item mappings
     * 
     * Maps user/item IDs to consecutive indices and initializes factor matrices.
     * 
     * @param ratings Training ratings
     */
    void initialize(const std::vector<Rating>& ratings) {
        std::mutex mtx;  // Mutex for thread-safe operations on containers
        
        // Create user and item ID mappings and count unique users and items
        std::unordered_map<int, int> userIdToIndex;
        std::unordered_map<int, int> itemIdToIndex;
        
        for (const auto& rating : ratings) {
            {
                std::lock_guard<std::mutex> lock(mtx);  // Thread-safe map access
                // Add user ID to map if not already present
                if (userIdToIndex.find(rating.getUserId()) == userIdToIndex.end()) {
                    userIdToIndex[rating.getUserId()] = userIdToIndex.size();
                }
                
                // Add item ID to map if not already present
                if (itemIdToIndex.find(rating.getItemId()) == itemIdToIndex.end()) {
                    itemIdToIndex[rating.getItemId()] = itemIdToIndex.size();
                }
            }
        }
        
        numUsers_ = userIdToIndex.size();
        numItems_ = itemIdToIndex.size();
        
        // Store mappings
        userIdToIndex_ = std::move(userIdToIndex);
        itemIdToIndex_ = std::move(itemIdToIndex);
        
        // Create reverse mappings for prediction
        indexToUserId_.resize(numUsers_);
        indexToItemId_.resize(numItems_);
        
        for (const auto& [userId, index] : userIdToIndex_) {
            indexToUserId_[index] = userId;
        }
        
        for (const auto& [itemId, index] : itemIdToIndex_) {
            indexToItemId_[index] = itemId;
        }
        
        // Initialize user and item factors with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 0.1);
        
        userFactors_.resize(numUsers_, std::vector<float>(numFactors_));
        itemFactors_.resize(numItems_, std::vector<float>(numFactors_));
        
        for (int u = 0; u < numUsers_; ++u) {
            for (int f = 0; f < numFactors_; ++f) {
                userFactors_[u][f] = dist(gen);
            }
        }
        
        for (int i = 0; i < numItems_; ++i) {
            for (int f = 0; f < numFactors_; ++f) {
                itemFactors_[i][f] = dist(gen);
            }
        }
        
        initialized_ = true;
    }
    
    /**
     * @brief Trains the model using stochastic gradient descent
     * 
     * @param ratings Training ratings
     * @return float Final RMSE on training data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    float train(const std::vector<Rating>& ratings) {
        if (!initialized_) {
            throw std::runtime_error("Model must be initialized before training");
        }
        
        // Training loop
        float finalRmse = 0.0;
        
        for (int iteration = 0; iteration < numIterations_; ++iteration) {
            float rmse = trainOneIteration(ratings);
            finalRmse = rmse;
            
            // Optional: Early stopping if improvement is minimal
            if (iteration > 0 && std::abs(rmse - prevRmse_) < 1e-4) {
                std::cout << "Early stopping at iteration " << iteration 
                          << " with RMSE: " << rmse << std::endl;
                break;
            }
            
            prevRmse_ = rmse;
            
            // Print progress every 10 iterations
            if ((iteration + 1) % 10 == 0 || iteration == 0) {
                std::cout << "Iteration " << (iteration + 1) 
                          << ", RMSE: " << rmse << std::endl;
            }
        }
        
        return finalRmse;
    }
    
    /**
     * @brief Predicts the rating for a given user-item pair
     * 
     * @param userId The ID of the user
     * @param itemId The ID of the item
     * @return float The predicted rating
     * @throws std::out_of_range if user or item is not in the training data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    float predict(int userId, int itemId) const {
        if (!initialized_) {
            throw std::runtime_error("Model must be initialized and trained before prediction");
        }
        
        // Get indices for user and item
        auto userIt = userIdToIndex_.find(userId);
        auto itemIt = itemIdToIndex_.find(itemId);
        
        if (userIt == userIdToIndex_.end()) {
            throw std::out_of_range("User ID not found in training data");
        }
        
        if (itemIt == itemIdToIndex_.end()) {
            throw std::out_of_range("Item ID not found in training data");
        }
        
        int userIndex = userIt->second;
        int itemIndex = itemIt->second;
        
        // Calculate dot product of user and item factors
        float prediction = 0.0;
        for (int f = 0; f < numFactors_; ++f) {
            prediction += userFactors_[userIndex][f] * itemFactors_[itemIndex][f];
        }
        
        // Clip prediction to valid rating range (e.g., 1-5)
        prediction = std::max(1.0f, std::min(5.0f, prediction));
        
        return prediction;
    }
    
    /**
     * @brief Gets the top N recommended items for a user
     * 
     * @param userId The ID of the user
     * @param n Number of recommendations to return
     * @param excludeRated Whether to exclude items the user has already rated
     * @param ratedItems Optional set of item IDs the user has already rated
     * @return std::vector<std::pair<int, float>> Vector of (itemId, predictedRating) pairs
     * @throws std::out_of_range if user is not in the training data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    std::vector<std::pair<int, float>> getTopNRecommendations(
            int userId, 
            int n = 10,
            bool excludeRated = true,
            const std::unordered_set<int>& ratedItems = std::unordered_set<int>()) const {
        
        if (!initialized_) {
            throw std::runtime_error("Model must be initialized and trained before recommendation");
        }
        
        auto userIt = userIdToIndex_.find(userId);
        if (userIt == userIdToIndex_.end()) {
            throw std::out_of_range("User ID not found in training data");
        }
        
        int userIndex = userIt->second;
        std::vector<std::pair<int, float>> recommendations;
        
        // Predict ratings for all items
        for (int itemIndex = 0; itemIndex < numItems_; ++itemIndex) {
            int itemId = indexToItemId_[itemIndex];
            
            // Skip items the user has already rated if excludeRated is true
            if (excludeRated && ratedItems.find(itemId) != ratedItems.end()) {
                continue;
            }
            
            // Calculate predicted rating
            float predictedRating = 0.0;
            for (int f = 0; f < numFactors_; ++f) {
                predictedRating += userFactors_[userIndex][f] * itemFactors_[itemIndex][f];
            }
            
            // Clip prediction to valid rating range
            predictedRating = std::max(1.0f, std::min(5.0f, predictedRating));
            
            recommendations.emplace_back(itemId, predictedRating);
        }
        
        // Sort recommendations by predicted rating (descending)
        std::sort(recommendations.begin(), recommendations.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top N recommendations
        if (n > 0 && static_cast<size_t>(n) < recommendations.size()) {
            recommendations.resize(n);
        }
        
        return recommendations;
    }
    
    /**
     * @brief Evaluates the model on test data
     * 
     * @param testRatings Test ratings
     * @return float RMSE on test data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    float evaluate(const std::vector<Rating>& testRatings) const {
        if (!initialized_) {
            throw std::runtime_error("Model must be initialized and trained before evaluation");
        }
        
        float sumSquaredError = 0.0;
        int count = 0;
        
        for (const auto& rating : testRatings) {
            try {
                float prediction = predict(rating.getUserId(), rating.getItemId());
                float error = prediction - rating.getValue();
                sumSquaredError += error * error;
                count++;
            } catch (const std::out_of_range&) {
                // Skip ratings for users or items not in training data
                continue;
            }
        }
        
        if (count == 0) {
            return 0.0;  // No valid predictions
        }
        
        return std::sqrt(sumSquaredError / count);
    }
    
    // Getters for model parameters
    int getNumUsers() const { return numUsers_; }
    int getNumItems() const { return numItems_; }
    int getNumFactors() const { return numFactors_; }

private:
    /**
     * @brief Performs one iteration of training using SGD
     * 
     * @param ratings Training ratings
     * @return float RMSE for this iteration
     */
    float trainOneIteration(const std::vector<Rating>& ratings) {
        float sumSquaredError = 0.0;
        
        // Process each rating
        for (const auto& rating : ratings) {
            int userId = rating.getUserId();
            int itemId = rating.getItemId();
            float actualRating = rating.getValue();
            
            // Get indices
            int userIndex = userIdToIndex_.at(userId);
            int itemIndex = itemIdToIndex_.at(itemId);
            
            // Calculate predicted rating
            float predictedRating = 0.0;
            for (int f = 0; f < numFactors_; ++f) {
                predictedRating += userFactors_[userIndex][f] * itemFactors_[itemIndex][f];
            }
            
            // Calculate error
            float error = predictedRating - actualRating;
            sumSquaredError += error * error;
            
            // Update user and item factors using gradient descent
            for (int f = 0; f < numFactors_; ++f) {
                // Cache old values before updating (needed for correct update)
                float oldUserFactor = userFactors_[userIndex][f];
                float oldItemFactor = itemFactors_[itemIndex][f];
                
                // Update user factor
                userFactors_[userIndex][f] -= learningRate_ * 
                                             (error * oldItemFactor + 
                                              regularization_ * oldUserFactor);
                
                // Update item factor
                itemFactors_[itemIndex][f] -= learningRate_ * 
                                             (error * oldUserFactor + 
                                              regularization_ * oldItemFactor);
            }
        }
        
        // Calculate RMSE
        return std::sqrt(sumSquaredError / ratings.size());
    }
    
    // Model hyperparameters
    int numFactors_;       // Number of latent factors
    float learningRate_;   // Learning rate for SGD
    float regularization_; // Regularization parameter
    int numIterations_;    // Number of training iterations
    
    // Model state
    int numUsers_;         // Number of unique users
    int numItems_;         // Number of unique items
    float prevRmse_;       // Previous iteration's RMSE (for early stopping)
    bool initialized_;     // Whether the model has been initialized
    
    // User and item factors (matrices)
    std::vector<std::vector<float>> userFactors_;  // User latent factors matrix
    std::vector<std::vector<float>> itemFactors_;  // Item latent factors matrix
    
    // Mappings between IDs and indices
    std::unordered_map<int, int> userIdToIndex_;   // Maps user IDs to matrix indices
    std::unordered_map<int, int> itemIdToIndex_;   // Maps item IDs to matrix indices
    std::vector<int> indexToUserId_;               // Maps matrix indices to user IDs
    std::vector<int> indexToItemId_;               // Maps matrix indices to item IDs
    
    // Mutex for thread safety
    mutable std::shared_mutex mutex_;  // Protects access to model parameters
};

/**
 * @class ThreadSafeRecommender
 * @brief Thread-safe wrapper for the MatrixFactorizationModel
 * 
 * Provides a thread-safe interface to the recommender model
 * using read-write locks (shared_mutex).
 */
class ThreadSafeRecommender {
public:
    /**
     * @brief Constructs a thread-safe recommender
     * 
     * @param numFactors Number of latent factors to use
     * @param learningRate Learning rate for gradient descent
     * @param regularization Regularization parameter to prevent overfitting
     * @param numIterations Number of training iterations
     */
    ThreadSafeRecommender(
            int numFactors = 10,
            float learningRate = 0.01,
            float regularization = 0.02,
            int numIterations = 100)
        : model_(numFactors, learningRate, regularization, numIterations) {}
    
    /**
     * @brief Initializes the model (thread-safe)
     * 
     * @param ratings Training ratings
     */
    void initialize(const std::vector<Rating>& ratings) {
        std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive lock for writing
        model_.initialize(ratings);
    }
    
    /**
     * @brief Trains the model (thread-safe)
     * 
     * @param ratings Training ratings
     * @return float Final RMSE on training data
     */
    float train(const std::vector<Rating>& ratings) {
        std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive lock for writing
        return model_.train(ratings);
    }
    
    /**
     * @brief Predicts a rating (thread-safe)
     * 
     * @param userId The ID of the user
     * @param itemId The ID of the item
     * @return float The predicted rating
     */
    float predict(int userId, int itemId) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);  // Shared lock for reading
        return model_.predict(userId, itemId);
    }
    
    /**
     * @brief Gets recommendations for a user (thread-safe)
     * 
     * @param userId The ID of the user
     * @param n Number of recommendations to return
     * @param excludeRated Whether to exclude items the user has already rated
     * @param ratedItems Optional set of item IDs the user has already rated
     * @return std::vector<std::pair<int, float>> Vector of (itemId, predictedRating) pairs
     */
    std::vector<std::pair<int, float>> getTopNRecommendations(
            int userId, 
            int n = 10,
            bool excludeRated = true,
            const std::unordered_set<int>& ratedItems = std::unordered_set<int>()) const {
        
        std::shared_lock<std::shared_mutex> lock(mutex_);  // Shared lock for reading
        return model_.getTopNRecommendations(userId, n, excludeRated, ratedItems);
    }
    
    /**
     * @brief Evaluates the model on test data (thread-safe)
     * 
     * @param testRatings Test ratings
     * @return float RMSE on test data
     */
    float evaluate(const std::vector<Rating>& testRatings) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);  // Shared lock for reading
        return model_.evaluate(testRatings);
    }

private:
    MatrixFactorizationModel model_;       // The underlying model
    mutable std::shared_mutex mutex_;      // Read-write lock for thread safety
                                          // (mutable so we can lock in const methods)
};

/**
 * @brief Example function to demonstrate usage of the recommender system
 */
void demonstrateRecommenderSystem() {
    // Create sample ratings data
    std::vector<Rating> ratings = {
        // User 1
        Rating(1, 101, 5.0),  // User 1 rates item 101 as 5.0
        Rating(1, 102, 3.0),  // User 1 rates item 102 as 3.0
        Rating(1, 103, 2.5),  // ...
        
        // User 2
        Rating(2, 101, 2.0),
        Rating(2, 102, 2.5),
        Rating(2, 104, 4.0),
        Rating(2, 105, 4.5),
        
        // User 3
        Rating(3, 101, 2.5),
        Rating(3, 102, 4.0),
        Rating(3, 103, 4.5),
        Rating(3, 104, 5.0),
        Rating(3, 105, 3.5),
        
        // User 4
        Rating(4, 101, 5.0),
        Rating(4, 103, 3.0),
        Rating(4, 104, 4.5),
        
        // User 5
        Rating(5, 102, 4.0),
        Rating(5, 103, 3.5),
        Rating(5, 104, 4.0),
        Rating(5, 105, 2.5)
    };
    
    // Create data loader and split data
    DataLoader dataLoader;
    auto [trainRatings, testRatings] = dataLoader.splitTrainTest(ratings, 0.2);
    
    std::cout << "Training set size: " << trainRatings.size() << std::endl;
    std::cout << "Test set size: " << testRatings.size() << std::endl;
    
    // Create and initialize recommender
    ThreadSafeRecommender recommender(10, 0.01, 0.02, 100);
    recommender.initialize(trainRatings);
    
    // Train the model
    std::cout << "Training the model..." << std::endl;
    float trainRmse = recommender.train(trainRatings);
    std::cout << "Final training RMSE: " << trainRmse << std::endl;
    
    // Evaluate on test data
    float testRmse = recommender.evaluate(testRatings);
    std::cout << "Test RMSE: " << testRmse << std::endl;
    
    // Generate recommendations for a user
    int userId = 1;
    int numRecommendations = 3;
    
    // Get items user 1 has already rated
    std::unordered_set<int> ratedItems;
    for (const auto& rating : trainRatings) {
        if (rating.getUserId() == userId) {
            ratedItems.insert(rating.getItemId());
        }
    }
    
    // Get recommendations
    auto recommendations = recommender.getTopNRecommendations(
        userId, numRecommendations, true, ratedItems);
    
    // Display recommendations
    std::cout << "Top " << numRecommendations << " recommendations for user " 
              << userId << ":" << std::endl;
    
    for (const auto& [itemId, predictedRating] : recommendations) {
        std::cout << "Item " << itemId << ": predicted rating = " 
                  << predictedRating << std::endl;
    }
    
    // Example of concurrent prediction (thread safety)
    std::cout << "\nTesting concurrent predictions..." << std::endl;
    
    auto predictFunction = [&recommender](int userId, int itemId) {
        try {
            float prediction = recommender.predict(userId, itemId);
            std::cout << "Predicted rating for user " << userId << ", item " 
                      << itemId << ": " << prediction << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    };
    
    // Launch multiple threads to test concurrent access
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(predictFunction, 1, 104);
        threads.emplace_back(predictFunction, 2, 103);
        threads.emplace_back(predictFunction, 3, 101);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

/**
 * @brief Main function
 */
int main() {
    try {
        std::cout << "Recommender System Demo" << std::endl;
        std::cout << "=======================" << std::endl;
        
        demonstrateRecommenderSystem();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}