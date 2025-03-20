# Code Overview: main.cpp

This C++ code implements the **K-Means clustering algorithm**, a popular unsupervised machine learning technique used for grouping data points into clusters based on their similarity. Let’s break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The purpose of this code is to:
1. **Group a set of 2D data points into clusters** based on their proximity to each other.
2. **Assign each data point to a cluster** such that points in the same cluster are more similar (closer in distance) to each other than to points in other clusters.
3. **Predict the cluster** for a new, unseen data point based on the learned cluster centroids.

This is a classic example of **clustering**, which is widely used in data analysis, pattern recognition, and machine learning for tasks like customer segmentation, image compression, and anomaly detection.

---

### **Main Functionality**
The code performs the following key tasks:
1. **Defines a 2D Point Structure**: Represents a data point with two features (`x1` and `x2`) and a cluster assignment.
2. **Computes Euclidean Distance**: Measures the distance between two points, which is used to determine similarity.
3. **Implements the K-Means Algorithm**:
   - Randomly initializes cluster centroids.
   - Iteratively assigns points to the nearest centroid and updates the centroids until convergence or a maximum number of iterations is reached.
4. **Predicts Clusters for New Points**: After clustering, the code can predict which cluster a new point belongs to based on the learned centroids.
5. **Handles User Input**: Allows the user to input a new point and see which cluster it is assigned to.

---

### **Algorithms Used**
1. **K-Means Clustering**:
   - **Input**: A set of data points and the number of clusters (`k`).
   - **Output**: Cluster assignments for each point and the centroids of the clusters.
   - **Steps**:
     - Randomly initialize `k` centroids.
     - Assign each point to the nearest centroid.
     - Recompute the centroids as the mean of all points in the cluster.
     - Repeat until centroids stop changing or the maximum number of iterations is reached.
2. **Euclidean Distance**:
   - Used to measure the distance between two points in 2D space.
   - Formula: `distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)`.

---

### **Overall Structure**
The code is organized into several functions and components that work together:

1. **Data Representation**:
   - The `Point` struct represents a 2D data point with features (`x1`, `x2`) and a cluster assignment (`cluster`).

2. **Distance Calculation**:
   - The `distance` function computes the Euclidean distance between two points.

3. **Centroid Calculation**:
   - The `compute_centroid` function calculates the mean (centroid) of all points in a given cluster.

4. **K-Means Algorithm**:
   - The `k_means` function implements the K-Means algorithm:
     - Randomly initializes centroids.
     - Assigns points to the nearest centroid.
     - Updates centroids iteratively until convergence.

5. **Prediction**:
   - The `predict_cluster` function assigns a new point to the nearest cluster based on the learned centroids.

6. **Main Function**:
   - Defines a hardcoded dataset.
   - Runs the K-Means algorithm.
   - Displays the clustering results.
   - Allows the user to input a new point and predict its cluster.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The `main` function defines a dataset of 2D points and sets the number of clusters (`k`).
   - The `k_means` function initializes the centroids randomly from the dataset.

2. **Clustering**:
   - The `k_means` function iteratively assigns points to the nearest centroid and updates the centroids until convergence or the maximum number of iterations is reached.

3. **Prediction**:
   - After clustering, the `predict_cluster` function is used to assign a new point to the nearest cluster based on the learned centroids.

4. **User Interaction**:
   - The user can input a new point, and the program predicts its cluster using the learned centroids.

---

### **Problem Being Solved**
The code solves the problem of **grouping similar data points into clusters**. This is useful in scenarios where you want to:
- Discover patterns or groupings in data.
- Reduce the complexity of data by representing it with cluster centroids.
- Make predictions about new data points based on existing clusters.

---

### **Approach Taken**
1. **Unsupervised Learning**:
   - The algorithm does not require labeled data. It groups points based on their inherent similarity.
2. **Iterative Optimization**:
   - The algorithm iteratively improves the cluster assignments and centroids until convergence.
3. **Distance-Based Similarity**:
   - Points are assigned to clusters based on their Euclidean distance to the centroids.

---

### **Key Components**
1. **Point Struct**:
   - Represents a data point with features and a cluster assignment.
2. **Distance Function**:
   - Computes the Euclidean distance between two points.
3. **Centroid Calculation**:
   - Computes the mean of all points in a cluster.
4. **K-Means Function**:
   - Implements the core clustering algorithm.
5. **Prediction Function**:
   - Assigns new points to clusters based on learned centroids.
6. **Main Function**:
   - Drives the program by defining the dataset, running the algorithm, and handling user input.

---

### **Example Use Case**
Suppose you have a dataset of customer locations (latitude and longitude). You can use this code to:
1. Group customers into clusters based on their proximity.
2. Identify the central location (centroid) of each cluster.
3. Predict which cluster a new customer location belongs to.

---

This code is a clear and concise implementation of the K-Means algorithm, designed to be both educational and practical. It demonstrates the core concepts of clustering and provides a foundation for further exploration in machine learning and data analysis.