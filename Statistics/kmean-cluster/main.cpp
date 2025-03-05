#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <ctime>
#include <chrono>

// Class to represent a data point in n-dimensional space
class Point {
private:
    std::vector<double> features;
    int cluster_id; // The ID of the cluster this point belongs to
    
public:
    // Constructor for an n-dimensional point
    Point(const std::vector<double>& features) 
        : features(features), cluster_id(-1) {}
    
    // Get the dimensionality of the point
    size_t getDimension() const {
        return features.size();
    }
    
    // Get a specific feature value
    double getFeature(size_t index) const {
        return features[index];
    }
    
    // Get all features
    const std::vector<double>& getFeatures() const {
        return features;
    }
    
    // Set the cluster ID for this point
    void setCluster(int id) {
        cluster_id = id;
    }
    
    // Get the cluster ID for this point
    int getCluster() const {
        return cluster_id;
    }
    
    // Calculate Euclidean distance between this point and another
    double distance(const Point& other) const {
        if (features.size() != other.features.size()) {
            throw std::runtime_error("Points have different dimensions");
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < features.size(); ++i) {
            double diff = features[i] - other.getFeature(i);
            sum += diff * diff;
        }
        
        return std::sqrt(sum);
    }
    
    // Print the point's features
    void print() const {
        std::cout << "(";
        for (size_t i = 0; i < features.size(); ++i) {
            std::cout << features[i];
            if (i < features.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ") -> Cluster: " << cluster_id;
    }
};

// K-Means clustering algorithm
class KMeans {
private:
    int k;                         // Number of clusters
    int max_iterations;            // Maximum iterations for convergence
    std::vector<Point> centroids;  // Cluster centroids
    
    // Initialize centroids using random points from the dataset
    void initializeCentroids(const std::vector<Point>& data) {
        // Use current time as seed for random generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> distrib(0, data.size() - 1);
        
        // Clear any existing centroids
        centroids.clear();
        
        // Track indices of chosen points to avoid duplicates
        std::vector<int> chosen_indices;
        
        // Select K random points as initial centroids
        while (centroids.size() < k && centroids.size() < data.size()) {
            int index = distrib(gen);
            
            // Check if this index has already been chosen
            if (std::find(chosen_indices.begin(), chosen_indices.end(), index) == chosen_indices.end()) {
                chosen_indices.push_back(index);
                centroids.push_back(data[index]);
                // Set the cluster ID of the centroid
                centroids.back().setCluster(centroids.size() - 1);
            }
        }
    }
    
    // Assign each point to the nearest centroid
    bool assignClusters(std::vector<Point>& data) {
        bool changed = false;
        
        // For each point
        for (auto& point : data) {
            // Find the nearest centroid
            double min_distance = std::numeric_limits<double>::max();
            int nearest_cluster = -1;
            
            for (size_t i = 0; i < centroids.size(); ++i) {
                double distance = point.distance(centroids[i]);
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_cluster = i;
                }
            }
            
            // If the cluster assignment changed, update it
            if (point.getCluster() != nearest_cluster) {
                point.setCluster(nearest_cluster);
                changed = true;
            }
        }
        
        return changed;
    }
    
    // Recalculate centroids based on the mean of all points in each cluster
    void updateCentroids(const std::vector<Point>& data) {
        if (data.empty()) return;
        
        size_t dimensions = data[0].getDimension();
        
        // For each cluster
        for (int i = 0; i < k; ++i) {
            // Count points in this cluster
            int cluster_size = 0;
            
            // Sum of feature values for calculating the mean
            std::vector<double> feature_sums(dimensions, 0.0);
            
            // Sum all feature values for points in this cluster
            for (const auto& point : data) {
                if (point.getCluster() == i) {
                    for (size_t d = 0; d < dimensions; ++d) {
                        feature_sums[d] += point.getFeature(d);
                    }
                    cluster_size++;
                }
            }
            
            // If the cluster is empty, leave the centroid as is
            if (cluster_size == 0) {
                std::cout << "Warning: Cluster " << i << " is empty!" << std::endl;
                continue;
            }
            
            // Calculate the mean for each feature
            std::vector<double> new_features;
            for (size_t d = 0; d < dimensions; ++d) {
                new_features.push_back(feature_sums[d] / cluster_size);
            }
            
            // Update the centroid
            if (i < centroids.size()) {
                centroids[i] = Point(new_features);
                centroids[i].setCluster(i);
            } else {
                Point new_centroid(new_features);
                new_centroid.setCluster(i);
                centroids.push_back(new_centroid);
            }
        }
    }
    
public:
    // Constructor
    KMeans(int k = 3, int max_iterations = 100) 
        : k(k), max_iterations(max_iterations) {
        if (k <= 0) {
            throw std::invalid_argument("Number of clusters must be positive");
        }
    }
    
    // Train the model on a dataset
    void fit(std::vector<Point>& data) {
        if (data.empty()) {
            throw std::runtime_error("Empty dataset");
        }
        
        if (k > data.size()) {
            std::cout << "Warning: k is larger than dataset size. Setting k = dataset size." << std::endl;
            k = data.size();
        }
        
        // Initialize centroids
        initializeCentroids(data);
        
        // Main K-Means loop
        bool changed = true;
        int iteration = 0;
        
        while (changed && iteration < max_iterations) {
            // Assign points to clusters
            changed = assignClusters(data);
            
            // Update centroids
            updateCentroids(data);
            
            iteration++;
        }
        
        std::cout << "K-Means converged after " << iteration << " iterations." << std::endl;
    }
    
    // Predict the cluster for a new point
    int predict(const Point& point) {
        if (centroids.empty()) {
            throw std::runtime_error("Model not trained yet");
        }
        
        // Find the nearest centroid
        double min_distance = std::numeric_limits<double>::max();
        int nearest_cluster = -1;
        
        for (size_t i = 0; i < centroids.size(); ++i) {
            double distance = point.distance(centroids[i]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_cluster = i;
            }
        }
        
        return nearest_cluster;
    }
    
    // Calculate the total within-cluster sum of squares (WCSS)
    // This is a measure of how tight the clusters are
    double calculateWCSS(const std::vector<Point>& data) {
        double wcss = 0.0;
        
        for (const auto& point : data) {
            int cluster_id = point.getCluster();
            if (cluster_id >= 0 && cluster_id < centroids.size()) {
                wcss += std::pow(point.distance(centroids[cluster_id]), 2);
            }
        }
        
        return wcss;
    }
    
    // Get the centroids
    const std::vector<Point>& getCentroids() const {
        return centroids;
    }
    
    // Print the centroids
    void printCentroids() const {
        std::cout << "Cluster Centroids:" << std::endl;
        for (size_t i = 0; i < centroids.size(); ++i) {
            std::cout << "Cluster " << i << ": ";
            centroids[i].print();
            std::cout << std::endl;
        }
    }
};

// Function to read customer data from a CSV file
std::vector<Point> readCustomerData(const std::string& filename) {
    std::vector<Point> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> features;
        
        // Skip customer ID
        std::getline(ss, cell, ',');
        
        // Read features (annual income and spending score)
        while (std::getline(ss, cell, ',')) {
            features.push_back(std::stod(cell));
        }
        
        if (!features.empty()) {
            data.push_back(Point(features));
        }
    }
    
    return data;
}

// Function to save the clustering results to a CSV file
void saveClusteringResults(const std::vector<Point>& data, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Feature1,Feature2,Cluster" << std::endl;
    
    // Write data
    for (const auto& point : data) {
        file << point.getFeature(0) << "," << point.getFeature(1) << "," << point.getCluster() << std::endl;
    }
    
    std::cout << "Results saved to " << filename << std::endl;
}

// Generate synthetic customer data for demonstration
std::vector<Point> generateCustomerData(int num_customers) {
    std::vector<Point> data;
    
    // Random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    
    // Income distribution (in thousands)
    std::normal_distribution<> income_dist(60.0, 20.0);
    
    // Spending score distribution (0-100)
    std::normal_distribution<> spending_dist(50.0, 15.0);
    
    // Create three distinct customer groups
    for (int i = 0; i < num_customers / 3; ++i) {
        // High income, high spending
        std::vector<double> features1 = {
            std::max(20.0, std::min(120.0, income_dist(gen) + 20.0)),
            std::max(0.0, std::min(100.0, spending_dist(gen) + 20.0))
        };
        data.push_back(Point(features1));
        
        // Medium income, medium spending
        std::vector<double> features2 = {
            std::max(20.0, std::min(120.0, income_dist(gen))),
            std::max(0.0, std::min(100.0, spending_dist(gen)))
        };
        data.push_back(Point(features2));
        
        // Low income, low spending
        std::vector<double> features3 = {
            std::max(20.0, std::min(120.0, income_dist(gen) - 20.0)),
            std::max(0.0, std::min(100.0, spending_dist(gen) - 20.0))
        };
        data.push_back(Point(features3));
    }
    
    return data;
}

// Function to normalize the data to [0,1] range
void normalizeData(std::vector<Point>& data) {
    if (data.empty()) return;
    
    size_t dimensions = data[0].getDimension();
    
    // Find min and max for each dimension
    std::vector<double> min_values(dimensions, std::numeric_limits<double>::max());
    std::vector<double> max_values(dimensions, std::numeric_limits<double>::lowest());
    
    for (const auto& point : data) {
        for (size_t d = 0; d < dimensions; ++d) {
            min_values[d] = std::min(min_values[d], point.getFeature(d));
            max_values[d] = std::max(max_values[d], point.getFeature(d));
        }
    }
    
    // Create normalized data
    std::vector<Point> normalized_data;
    
    for (const auto& point : data) {
        std::vector<double> normalized_features;
        
        for (size_t d = 0; d < dimensions; ++d) {
            double range = max_values[d] - min_values[d];
            // Avoid division by zero
            if (range == 0) {
                normalized_features.push_back(0.5); // Map to middle of range
            } else {
                normalized_features.push_back((point.getFeature(d) - min_values[d]) / range);
            }
        }
        
        normalized_data.push_back(Point(normalized_features));
    }
    
    // Replace original data with normalized data
    data = std::move(normalized_data);
}

// Example application: Customer segmentation
int main() {
    // Generate synthetic customer data (150 customers with annual income and spending score)
    std::vector<Point> customers = generateCustomerData(150);
    
    std::cout << "Generated data for " << customers.size() << " customers." << std::endl;
    std::cout << "Each customer has 2 features: Annual Income and Spending Score" << std::endl;
    
    // Print first 5 customers
    std::cout << "\nSample customer data (first 5):" << std::endl;
    for (size_t i = 0; i < 5 && i < customers.size(); ++i) {
        std::cout << "Customer " << i << ": ";
        customers[i].print();
        std::cout << std::endl;
    }
    
    // Normalize the data
    std::cout << "\nNormalizing data..." << std::endl;
    normalizeData(customers);
    
    // Create K-Means model with k=3 clusters
    int num_clusters = 3;
    KMeans kmeans(num_clusters);
    
    // Train the model
    std::cout << "\nTraining K-Means model with " << num_clusters << " clusters..." << std::endl;
    kmeans.fit(customers);
    
    // Print the cluster centroids
    std::cout << "\nFinal cluster centroids:" << std::endl;
    kmeans.printCentroids();
    
    // Calculate the WCSS
    double wcss = kmeans.calculateWCSS(customers);
    std::cout << "\nWithin-Cluster Sum of Squares (WCSS): " << wcss << std::endl;
    
    // Count the number of customers in each cluster
    std::vector<int> cluster_counts(num_clusters, 0);
    for (const auto& customer : customers) {
        cluster_counts[customer.getCluster()]++;
    }
    
    // Print cluster statistics
    std::cout << "\nCluster statistics:" << std::endl;
    for (int i = 0; i < num_clusters; ++i) {
        std::cout << "Cluster " << i << ": " << cluster_counts[i] << " customers" << std::endl;
    }
    
    // Predict the cluster for a new customer
    std::vector<double> new_customer_features = {0.7, 0.8}; // Normalized values
    Point new_customer(new_customer_features);
    
    int predicted_cluster = kmeans.predict(new_customer);
    std::cout << "\nNew customer (high income, high spending) predicted to be in cluster: " 
              << predicted_cluster << std::endl;
    
    // Save the results to a CSV file
    saveClusteringResults(customers, "customer_clusters.csv");
    
    std::cout << "\nCustomer segmentation complete!" << std::endl;
    
    return 0;
}