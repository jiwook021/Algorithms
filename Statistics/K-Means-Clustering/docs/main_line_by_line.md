# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step** in a way that’s accessible to everyone, from beginners to experts. I’ll explain each section in detail, define technical terms, and provide examples to make everything clear.

---

### **1. Includes and Libraries**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
```
#### What it does:
- These lines include libraries that provide functionality for:
  - **Input/Output** (`iostream`): For printing to the console and reading user input.
  - **Vectors** (`vector`): For storing collections of data (like lists of points).
  - **Math Functions** (`cmath`): For mathematical operations like square roots and powers.
  - **Random Number Generation** (`cstdlib` and `ctime`): For generating random numbers.

#### Why it’s used:
- These libraries are essential for the program to perform tasks like printing results, storing data, and calculating distances.

---

### **2. Point Structure**
```cpp
struct Point {
    double x1, x2;  // Features
    int cluster;    // Assigned cluster (-1 means unassigned)
};
```
#### What it does:
- Defines a **structure** (a custom data type) called `Point`.
- Each `Point` has:
  - Two features (`x1` and `x2`): These represent the coordinates of the point in 2D space.
  - A `cluster` value: This stores which cluster the point belongs to. Initially, it’s set to `-1` (unassigned).

#### Why it’s used:
- Structures allow us to group related data together. Here, we group the coordinates and cluster assignment for each point.

#### Example:
- A `Point` with `x1 = 1.0`, `x2 = 2.0`, and `cluster = -1` represents a point at coordinates (1.0, 2.0) that hasn’t been assigned to a cluster yet.

---

### **3. Distance Function**
```cpp
double distance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x1 - p2.x1, 2) + std::pow(p1.x2 - p2.x2, 2));
}
```
#### What it does:
- Computes the **Euclidean distance** between two points (`p1` and `p2`).
- The formula is:
  ```
  distance = sqrt((x1_diff)^2 + (x2_diff)^2)
  ```
  where `x1_diff` is the difference in `x1` values, and `x2_diff` is the difference in `x2` values.

#### Why it’s used:
- Euclidean distance measures how "far apart" two points are. This is used to determine which cluster a point should belong to.

#### Example:
- If `p1` is (1.0, 2.0) and `p2` is (4.0, 6.0):
  ```
  distance = sqrt((1.0 - 4.0)^2 + (2.0 - 6.0)^2)
            = sqrt(9 + 16)
            = sqrt(25)
            = 5.0
  ```

---

### **4. Centroid Calculation**
```cpp
Point compute_centroid(const std::vector<Point>& points, int cluster) {
    double sum_x1 = 0.0, sum_x2 = 0.0;
    int count = 0;
    for (const auto& p : points) {
        if (p.cluster == cluster) {
            sum_x1 += p.x1;
            sum_x2 += p.x2;
            count++;
        }
    }
    if (count == 0) return {0.0, 0.0, cluster};  // Avoid division by zero
    return {sum_x1 / count, sum_x2 / count, cluster};
}
```
#### What it does:
- Computes the **centroid** (average point) for a given cluster.
- Steps:
  1. Sums up the `x1` and `x2` values of all points in the cluster.
  2. Divides the sums by the number of points in the cluster to get the average.
  3. Returns the centroid as a `Point`.

#### Why it’s used:
- Centroids represent the "center" of a cluster. Updating centroids is a key step in the K-Means algorithm.

#### Example:
- If a cluster has points (1.0, 2.0), (2.0, 3.0), and (3.0, 4.0):
  ```
  sum_x1 = 1.0 + 2.0 + 3.0 = 6.0
  sum_x2 = 2.0 + 3.0 + 4.0 = 9.0
  count = 3
  centroid = (6.0 / 3, 9.0 / 3) = (2.0, 3.0)
  ```

---

### **5. K-Means Algorithm**
```cpp
void k_means(std::vector<Point>& data, int k, int max_iterations = 100) {
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Initialize centroids randomly from data points
    std::vector<Point> centroids;
    for (int i = 0; i < k; ++i) {
        int idx = std::rand() % data.size();
        centroids.push_back({data[idx].x1, data[idx].x2, i});
    }

    bool changed = true;
    int iterations = 0;

    while (changed && iterations < max_iterations) {
        changed = false;
        iterations++;

        // Step 1: Assign points to nearest centroid
        for (auto& point : data) {
            double min_dist = distance(point, centroids[0]);
            int new_cluster = 0;

            for (int c = 1; c < k; ++c) {
                double dist = distance(point, centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    new_cluster = c;
                }
            }

            if (point.cluster != new_cluster) {
                point.cluster = new_cluster;
                changed = true;
            }
        }

        // Step 2: Update centroids
        for (int c = 0; c < k; ++c) {
            centroids[c] = compute_centroid(data, c);
        }
    }

    std::cout << "Converged after " << iterations << " iterations.\n";
}
```
#### What it does:
- Implements the **K-Means clustering algorithm**:
  1. Randomly initializes `k` centroids.
  2. Assigns each point to the nearest centroid.
  3. Updates the centroids based on the new cluster assignments.
  4. Repeats steps 2 and 3 until the centroids stop changing or the maximum number of iterations is reached.

#### Why it’s used:
- K-Means is a simple and effective algorithm for clustering data.

#### Example:
- Suppose `k = 2` and the dataset has points (1.0, 2.0), (2.0, 3.0), (8.0, 8.0), and (9.0, 9.0).
  1. Randomly initialize centroids, e.g., (1.0, 2.0) and (8.0, 8.0).
  2. Assign points to the nearest centroid:
     - (1.0, 2.0) and (2.0, 3.0) are closer to (1.0, 2.0).
     - (8.0, 8.0) and (9.0, 9.0) are closer to (8.0, 8.0).
  3. Update centroids:
     - New centroid for cluster 0: ((1.0 + 2.0)/2, (2.0 + 3.0)/2) = (1.5, 2.5).
     - New centroid for cluster 1: ((8.0 + 9.0)/2, (8.0 + 9.0)/2) = (8.5, 8.5).
  4. Repeat until centroids stop changing.

---

### **6. Prediction Function**
```cpp
int predict_cluster(const Point& new_point, const std::vector<Point>& centroids) {
    double min_dist = distance(new_point, centroids[0]);
    int cluster = 0;

    for (int c = 1; c < static_cast<int>(centroids.size()); ++c) {
        double dist = distance(new_point, centroids[c]);
        if (dist < min_dist) {
            min_dist = dist;
            cluster = c;
        }
    }
    return cluster;
}
```
#### What it does:
- Predicts the cluster for a new point by finding the nearest centroid.

#### Why it’s used:
- After clustering, we can use the learned centroids to classify new points.

#### Example:
- If the centroids are (1.5, 2.5) and (8.5, 8.5), and the new point is (2.0, 3.0):
  ```
  Distance to centroid 0: sqrt((2.0 - 1.5)^2 + (3.0 - 2.5)^2) = sqrt(0.25 + 0.25) = 0.707
  Distance to centroid 1: sqrt((2.0 - 8.5)^2 + (3.0 - 8.5)^2) = sqrt(42.25 + 30.25) = 8.5
  ```
  The new point is assigned to cluster 0 because it’s closer to centroid 0.

---

### **7. Main Function**
```cpp
int main() {
    // Hardcoded dataset
    std::vector<Point> dataset = {
        {1.0, 2.0, -1},
        {1.5, 1.8, -1},
        {5.0, 8.0, -1},
        {8.0, 8.0, -1},
        {1.0, 0.6, -1},
        {9.0, 11.0, -1}
    };

    // Number of clusters
    int k = 4;

    // Run K-Means
    k_means(dataset, k);

    // Display results
    std::cout << "Final clustering:\n";
    for (const auto& p : dataset) {
        std::cout << "Point (" << p.x1 << ", " << p.x2 << ") -> Cluster " << p.cluster << "\n";
    }

    // Store final centroids
    std::vector<Point> centroids(k);
    for (int c = 0; c < k; ++c) {
        centroids[c] = compute_centroid(dataset, c);
    }

    // User input for prediction
    std::cout << "\nEnter x1 and x2 to predict cluster (e.g., 3.0 4.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;
    Point new_point = {x1, x2, -1};

    // Predict cluster for new point
    int pred_cluster = predict_cluster(new_point, centroids);
    std::cout << "Predicted cluster for (" << x1 << ", " << x2 << "): " << pred_cluster << "\n";

    return 0;
}
```
#### What it does:
- Defines a dataset, runs the K-Means algorithm, displays the results, and allows the user to predict the cluster for a new point.

#### Why it’s used:
- This is the entry point of the program. It ties everything together and provides a user-friendly interface.

#### Example:
- If the user inputs `3.0 4.0`, the program will predict which cluster the point (3.0, 4.0) belongs to based on the learned centroids.

---

This step-by-step explanation should make the code accessible to everyone, regardless of their programming experience! Let me know if you’d like further clarification on any part.