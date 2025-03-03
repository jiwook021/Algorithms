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
    Rating(int UserId, int ItemId, float value) 
        : UserId_(UserId), ItemId_(ItemId), value_(value) {}
    
    int GetUserId() const { return UserId_; }
    int GetItemId() const { return ItemId_; }
    float GetValue() const { return value_; }

private:
    int UserId_;    // The ID of the user who provided the rating
    int ItemId_;    // The ID of the item that was rated
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
    std::vector<Rating> LoadRatingsFromCsv(const std::string& Filename, bool HasHeader = true) {
        std::vector<Rating> Ratings;
        std::ifstream File(Filename);
        
        if (!File.is_open()) {
            throw std::runtime_error("Could not open file: " + Filename);
        }
        
        std::string Line;
        
        // Skip header if present
        if (HasHeader && std::getline(File, Line)) {
            // Do nothing, just skipping the header
        }
        
        while (std::getline(File, Line)) {
            std::stringstream Ss(Line);
            std::string Cell;
            std::vector<std::string> Cells;
            
            while (std::getline(Ss, Cell, ',')) {
                Cells.push_back(Cell);
            }
            
            if (Cells.size() < 3) {
                throw std::runtime_error("Invalid CSV format: Each line must contain at least userId, itemId, and rating");
            }
            
            try {
                int UserId = std::stoi(Cells[0]);
                int ItemId = std::stoi(Cells[1]);
                float RatingValue = std::stof(Cells[2]);
                
                Ratings.emplace_back(UserId, ItemId, RatingValue);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing values from CSV: " + std::string(e.what()));
            }
        }
        
        return Ratings;
    }
    
    /**
     * @brief Splits ratings into training and test sets
     * 
     * @param ratings The complete set of ratings
     * @param testFraction Fraction of ratings to use for testing (0-1)
     * @param seed Random seed for reproducibility
     * @return std::pair<std::vector<Rating>, std::vector<Rating>> Training and test sets
     */
    std::pair<std::vector<Rating>, std::vector<Rating>> SplitTrainTest(
            const std::vector<Rating>& Ratings, 
            float TestFraction = 0.2,
            unsigned int Seed = 42) {
        
        if (TestFraction < 0.0 || TestFraction > 1.0) {
            throw std::invalid_argument("testFraction must be between 0 and 1");
        }
        
        // Create a copy of the ratings that we can shuffle
        std::vector<Rating> RatingsCopy = Ratings;
        
        // Shuffle the ratings
        std::mt19937 Rng(Seed);
        std::shuffle(RatingsCopy.begin(), RatingsCopy.end(), Rng);
        
        // Calculate split point
        size_t TestSize = static_cast<size_t>(RatingsCopy.size() * TestFraction);
        size_t TrainSize = RatingsCopy.size() - TestSize;
        
        // Split the data
        std::vector<Rating> TrainSet(RatingsCopy.begin(), RatingsCopy.begin() + TrainSize);
        std::vector<Rating> TestSet(RatingsCopy.begin() + TrainSize, RatingsCopy.end());
        
        return {TrainSet, TestSet};
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
            int NumFactors = 10,
            float LearningRate = 0.01,
            float Regularization = 0.02,
            int NumIterations = 100)
        : NumFactors_(NumFactors),
          LearningRate_(LearningRate),
          Regularization_(Regularization),
          NumIterations_(NumIterations),
          Initialized_(false) {}
    
    /**
     * @brief Initializes the model with user and item mappings
     * 
     * Maps user/item IDs to consecutive indices and initializes factor matrices.
     * 
     * @param ratings Training ratings
     */
    void Initialize(const std::vector<Rating>& Ratings) {
        std::mutex Mtx;  // Mutex for thread-safe operations on containers
        
        // Create user and item ID mappings and count unique users and items
        std::unordered_map<int, int> UserIdToIndex;
        std::unordered_map<int, int> ItemIdToIndex;
        
        for (const auto& Rating : Ratings) {
            {
                std::lock_guard<std::mutex> Lock(Mtx);  // Thread-safe map access
                // Add user ID to map if not already present
                if (UserIdToIndex.find(Rating.GetUserId()) == UserIdToIndex.end()) {
                    UserIdToIndex[Rating.GetUserId()] = UserIdToIndex.size();
                }
                
                // Add item ID to map if not already present
                if (ItemIdToIndex.find(Rating.GetItemId()) == ItemIdToIndex.end()) {
                    ItemIdToIndex[Rating.GetItemId()] = ItemIdToIndex.size();
                }
            }
        }
        
        NumUsers_ = UserIdToIndex.size();
        NumItems_ = ItemIdToIndex.size();
        
        // Store mappings
        UserIdToIndex_ = std::move(UserIdToIndex);
        ItemIdToIndex_ = std::move(ItemIdToIndex);
        
        // Create reverse mappings for prediction
        IndexToUserId_.resize(NumUsers_);
        IndexToItemId_.resize(NumItems_);
        
        for (const auto& [UserId, Index] : UserIdToIndex_) {
            IndexToUserId_[Index] = UserId;
        }
        
        for (const auto& [ItemId, Index] : ItemIdToIndex_) {
            IndexToItemId_[Index] = ItemId;
        }
        
        // Initialize user and item factors with small random values
        std::random_device Rd;
        std::mt19937 Gen(Rd());
        std::uniform_real_distribution<float> Dist(0.0, 0.1);
        
        UserFactors_.resize(NumUsers_, std::vector<float>(NumFactors_));
        ItemFactors_.resize(NumItems_, std::vector<float>(NumFactors_));
        
        for (int u = 0; u < NumUsers_; ++u) {
            for (int f = 0; f < NumFactors_; ++f) {
                UserFactors_[u][f] = Dist(Gen);
            }
        }
        
        for (int i = 0; i < NumItems_; ++i) {
            for (int f = 0; f < NumFactors_; ++f) {
                ItemFactors_[i][f] = Dist(Gen);
            }
        }
        
        Initialized_ = true;
    }
    
    /**
     * @brief Trains the model using stochastic gradient descent
     * 
     * @param ratings Training ratings
     * @return float Final RMSE on training data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    float Train(const std::vector<Rating>& Ratings) {
        if (!Initialized_) {
            throw std::runtime_error("Model must be initialized before training");
        }
        
        // Training loop
        float FinalRmse = 0.0;
        
        for (int Iteration = 0; Iteration < NumIterations_; ++Iteration) {
            float Rmse = TrainOneIteration(Ratings);
            FinalRmse = Rmse;
            
            // Optional: Early stopping if improvement is minimal
            if (Iteration > 0 && std::abs(Rmse - PrevRmse_) < 1e-4) {
                std::cout << "Early stopping at iteration " << Iteration 
                          << " with RMSE: " << Rmse << std::endl;
                break;
            }
            
            PrevRmse_ = Rmse;
            
            // Print progress every 10 iterations
            if ((Iteration + 1) % 10 == 0 || Iteration == 0) {
                std::cout << "Iteration " << (Iteration + 1) 
                          << ", RMSE: " << Rmse << std::endl;
            }
        }
        
        return FinalRmse;
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
    float Predict(int UserId, int ItemId) const {
        if (!Initialized_) {
            throw std::runtime_error("Model must be initialized and trained before prediction");
        }
        
        // Get indices for user and item
        auto UserIt = UserIdToIndex_.find(UserId);
        auto ItemIt = ItemIdToIndex_.find(ItemId);
        
        if (UserIt == UserIdToIndex_.end()) {
            throw std::out_of_range("User ID not found in training data");
        }
        
        if (ItemIt == ItemIdToIndex_.end()) {
            throw std::out_of_range("Item ID not found in training data");
        }
        
        int UserIndex = UserIt->second;
        int ItemIndex = ItemIt->second;
        
        // Calculate dot product of user and item factors
        float Prediction = 0.0;
        for (int f = 0; f < NumFactors_; ++f) {
            Prediction += UserFactors_[UserIndex][f] * ItemFactors_[ItemIndex][f];
        }
        
        // Clip prediction to valid rating range (e.g., 1-5)
        Prediction = std::max(1.0f, std::min(5.0f, Prediction));
        
        return Prediction;
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
    std::vector<std::pair<int, float>> GetTopNRecommendations(
            int UserId, 
            int n = 10,
            bool ExcludeRated = true,
            const std::unordered_set<int>& RatedItems = std::unordered_set<int>()) const {
        
        if (!Initialized_) {
            throw std::runtime_error("Model must be initialized and trained before recommendation");
        }
        
        auto UserIt = UserIdToIndex_.find(UserId);
        if (UserIt == UserIdToIndex_.end()) {
            throw std::out_of_range("User ID not found in training data");
        }
        
        int UserIndex = UserIt->second;
        std::vector<std::pair<int, float>> Recommendations;
        
        // Predict ratings for all items
        for (int ItemIndex = 0; ItemIndex < NumItems_; ++ItemIndex) {
            int ItemId = IndexToItemId_[ItemIndex];
            
            // Skip items the user has already rated if excludeRated is true
            if (ExcludeRated && RatedItems.find(ItemId) != RatedItems.end()) {
                continue;
            }
            
            // Calculate predicted rating
            float PredictedRating = 0.0;
            for (int f = 0; f < NumFactors_; ++f) {
                PredictedRating += UserFactors_[UserIndex][f] * ItemFactors_[ItemIndex][f];
            }
            
            // Clip prediction to valid rating range
            PredictedRating = std::max(1.0f, std::min(5.0f, PredictedRating));
            
            Recommendations.emplace_back(ItemId, PredictedRating);
        }
        
        // Sort recommendations by predicted rating (descending)
        std::sort(Recommendations.begin(), Recommendations.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top N recommendations
        if (n > 0 && static_cast<size_t>(n) < Recommendations.size()) {
            Recommendations.resize(n);
        }
        
        return Recommendations;
    }
    
    /**
     * @brief Evaluates the model on test data
     * 
     * @param testRatings Test ratings
     * @return float RMSE on test data
     * @throws std::runtime_error if the model hasn't been initialized
     */
    float Evaluate(const std::vector<Rating>& TestRatings) const {
        if (!Initialized_) {
            throw std::runtime_error("Model must be initialized and trained before evaluation");
        }
        
        float SumSquaredError = 0.0;
        int count = 0;
        
        for (const auto& Rating : TestRatings) {
            try {
                float Prediction = Predict(Rating.GetUserId(), Rating.GetItemId());
                float Error = Prediction - Rating.GetValue();
                SumSquaredError += Error * Error;
                count++;
            } catch (const std::out_of_range&) {
                // Skip ratings for users or items not in training data
                continue;
            }
        }
        
        if (count == 0) {
            return 0.0;  // No valid predictions
        }
        
        return std::sqrt(SumSquaredError / count);
    }
    
    // Getters for model parameters
    int GetNumUsers() const { return NumUsers_; }
    int GetNumItems() const { return NumItems_; }
    int GetNumFactors() const { return NumFactors_; }

private:
    /**
     * @brief Performs one iteration of training using SGD
     * 
     * @param ratings Training ratings
     * @return float RMSE for this iteration
     */
    float TrainOneIteration(const std::vector<Rating>& Ratings) {
        float SumSquaredError = 0.0;
        
        // Process each rating
        for (const auto& Rating : Ratings) {
            int UserId = Rating.GetUserId();
            int ItemId = Rating.GetItemId();
            float ActualRating = Rating.GetValue();
            
            // Get indices
            int UserIndex = UserIdToIndex_.at(UserId);
            int ItemIndex = ItemIdToIndex_.at(ItemId);
            
            // Calculate predicted rating
            float PredictedRating = 0.0;
            for (int f = 0; f < NumFactors_; ++f) {
                PredictedRating += UserFactors_[UserIndex][f] * ItemFactors_[ItemIndex][f];
            }
            
            // Calculate error
            float Error = PredictedRating - ActualRating;
            SumSquaredError += Error * Error;
            
            // Update user and item factors using gradient descent
            for (int f = 0; f < NumFactors_; ++f) {
                // Cache old values before updating (needed for correct update)
                float OldUserFactor = UserFactors_[UserIndex][f];
                float OldItemFactor = ItemFactors_[ItemIndex][f];
                
                // Update user factor
                UserFactors_[UserIndex][f] -= LearningRate_ * 
                                             (Error * OldItemFactor + 
                                              Regularization_ * OldUserFactor);
                
                // Update item factor
                ItemFactors_[ItemIndex][f] -= LearningRate_ * 
                                             (Error * OldUserFactor + 
                                              Regularization_ * OldItemFactor);
            }
        }
        
        // Calculate RMSE
        return std::sqrt(SumSquaredError / Ratings.size());
    }
    
    // Model hyperparameters
    int NumFactors_;       // Number of latent factors
    float LearningRate_;   // Learning rate for SGD
    float Regularization_; // Regularization parameter
    int NumIterations_;    // Number of training iterations
    
    // Model state
    int NumUsers_;         // Number of unique users
    int NumItems_;         // Number of unique items
    float PrevRmse_;       // Previous iteration's RMSE (for early stopping)
    bool Initialized_;     // Whether the model has been initialized
    
    // User and item factors (matrices)
    std::vector<std::vector<float>> UserFactors_;  // User latent factors matrix
    std::vector<std::vector<float>> ItemFactors_;  // Item latent factors matrix
    
    // Mappings between IDs and indices
    std::unordered_map<int, int> UserIdToIndex_;   // Maps user IDs to matrix indices
    std::unordered_map<int, int> ItemIdToIndex_;   // Maps item IDs to matrix indices
    std::vector<int> IndexToUserId_;               // Maps matrix indices to user IDs
    std::vector<int> IndexToItemId_;               // Maps matrix indices to item IDs
    
    // Mutex for thread safety
    mutable std::shared_mutex Mutex_;  // Protects access to model parameters
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
            int NumFactors = 10,
            float LearningRate = 0.01,
            float Regularization = 0.02,
            int NumIterations = 100)
        : Model_(NumFactors, LearningRate, Regularization, NumIterations) {}
    
    /**
     * @brief Initializes the model (thread-safe)
     * 
     * @param ratings Training ratings
     */
    void Initialize(const std::vector<Rating>& Ratings) {
        std::unique_lock<std::shared_mutex> Lock(Mutex_);  // Exclusive lock for writing
        Model_.Initialize(Ratings);
    }
    
    /**
     * @brief Trains the model (thread-safe)
     * 
     * @param ratings Training ratings
     * @return float Final RMSE on training data
     */
    float Train(const std::vector<Rating>& Ratings) {
        std::unique_lock<std::shared_mutex> Lock(Mutex_);  // Exclusive lock for writing
        return Model_.Train(Ratings);
    }
    
    /**
     * @brief Predicts a rating (thread-safe)
     * 
     * @param userId The ID of the user
     * @param itemId The ID of the item
     * @return float The predicted rating
     */
    float Predict(int UserId, int ItemId) const {
        std::shared_lock<std::shared_mutex> lock(Mutex_);  // Shared lock for reading
        return Model_.Predict(UserId, ItemId);
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
    std::vector<std::pair<int, float>> GetTopNRecommendations(
            int UserId, 
            int n = 10,
            bool ExcludeRated = true,
            const std::unordered_set<int>& RatedItems = std::unordered_set<int>()) const {
        
        std::shared_lock<std::shared_mutex> lock(Mutex_);  // Shared lock for reading
        return Model_.GetTopNRecommendations(UserId, n, ExcludeRated, RatedItems);
    }
    
    /**
     * @brief Evaluates the model on test data (thread-safe)
     * 
     * @param testRatings Test ratings
     * @return float RMSE on test data
     */
    float Evaluate(const std::vector<Rating>& TestRatings) const {
        std::shared_lock<std::shared_mutex> lock(Mutex_);  // Shared lock for reading
        return Model_.Evaluate(TestRatings);
    }

private:
    MatrixFactorizationModel Model_;       // The underlying model
    mutable std::shared_mutex Mutex_;      // Read-write lock for thread safety
                                          // (mutable so we can lock in const methods)
};

/**
 * @brief Example function to demonstrate usage of the recommender system
 */
void DemonstrateRecommenderSystem() {
    // Create sample ratings data
    std::vector<Rating> Ratings = {
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
    DataLoader DataLoader;
    auto [TrainRatings, TestRatings] = DataLoader.SplitTrainTest(Ratings, 0.2);
    
    std::cout << "Training set size: " << TrainRatings.size() << std::endl;
    std::cout << "Test set size: " << TestRatings.size() << std::endl;
    
    // Create and initialize recommender
    ThreadSafeRecommender Recommender(10, 0.01, 0.02, 100);
    Recommender.Initialize(TrainRatings);
    
    // Train the model
    std::cout << "Training the model..." << std::endl;
    float TrainRmse = Recommender.Train(TrainRatings);
    std::cout << "Final training RMSE: " << TrainRmse << std::endl;
    
    // Evaluate on test data
    float TestRmse = Recommender.Evaluate(TestRatings);
    std::cout << "Test RMSE: " << TestRmse << std::endl;
    
    // Generate recommendations for a user
    int UserId = 1;
    int NumRecommendations = 3;
    
    // Get items user 1 has already rated
    std::unordered_set<int> RatedItems;
    for (const auto& Rating : TrainRatings) {
        if (Rating.GetUserId() == UserId) {
            RatedItems.insert(Rating.GetItemId());
        }
    }
    
    // Get recommendations
    auto Recommendations = Recommender.GetTopNRecommendations(
        UserId, NumRecommendations, true, RatedItems);
    
    // Display recommendations
    std::cout << "Top " << NumRecommendations << " recommendations for user " 
              << UserId << ":" << std::endl;
    
    for (const auto& [ItemId, PredictedRating] : Recommendations) {
        std::cout << "Item " << ItemId << ": predicted rating = " 
                  << PredictedRating << std::endl;
    }
    
    // Example of concurrent prediction (thread safety)
    std::cout << "\nTesting concurrent predictions..." << std::endl;
    
    auto PredictFunction = [&Recommender](int UserId, int ItemId) {
        try {
            float Prediction = Recommender.Predict(UserId, ItemId);
            std::cout << "Predicted rating for user " << UserId << ", item " 
                      << ItemId << ": " << Prediction << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    };
    
    // Launch multiple threads to test concurrent access
    std::vector<std::thread> Threads;
    for (int i = 0; i < 5; ++i) {
        Threads.emplace_back(PredictFunction, 1, 104);
        Threads.emplace_back(PredictFunction, 2, 103);
        Threads.emplace_back(PredictFunction, 3, 101);
    }
    
    // Wait for all threads to complete
    for (auto& thread : Threads) {
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
        
        DemonstrateRecommenderSystem();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}