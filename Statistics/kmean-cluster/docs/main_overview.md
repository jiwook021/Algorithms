# Code Overview: main.cpp

This code implements a **K-Means clustering algorithm** in C++ to group data points into clusters based on their similarity. Let's break down the purpose, functionality, and structure of the code in detail:

---

### **Purpose of the Code**
The code is designed to perform **unsupervised machine learning** by grouping a set of data points into clusters. Specifically:
1. It takes a dataset of points in n-dimensional space (e.g., customer data with features like annual income and spending score).
2. It groups these points into `k` clusters, where `k` is a user-defined number of clusters.
3. The goal is to minimize the distance between points within the same cluster (intra-cluster distance) while maximizing the distance between points in different clusters (inter-cluster distance).

This is a common technique used in **customer segmentation**, **image compression**, **anomaly detection**, and other data analysis tasks.

---

### **Main Functionality**
The code consists of two main components:
1. **`Point` Class**: Represents a data point in n-dimensional space.
2. **`KMeans` Class**: Implements the K-Means clustering algorithm.

The program follows these steps:
1. **Generate or load data**: The code generates synthetic customer data (e.g., annual income and spending score) for demonstration purposes.
2. **Normalize the data**: Scales the data to ensure all features contribute equally to the clustering process.
3. **Initialize K-Means**: Sets up the K-Means algorithm with a specified number of clusters (`k`).
4. **Train the model**: Runs the K-Means algorithm to assign each data point to a cluster.
5. **Output results**: Displays the cluster assignments for each data point.

---

### **Algorithms Used**
1. **K-Means Clustering**:
   - A centroid-based clustering algorithm.
   - Works by iteratively:
     - Assigning each data point to the nearest centroid.
     - Updating the centroids to the mean of all points in the cluster.
   - Continues until centroids stabilize or a maximum number of iterations is reached.

2. **Euclidean Distance**:
   - Used to measure the distance between two points in n-dimensional space.
   - Formula: 
     \[
     \text{distance}(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
     \]
   - This distance metric determines which cluster a point belongs to.

3. **Data Normalization**:
   - Scales the data so that all features have the same range (e.g., 0 to 1).
   - Ensures that no single feature dominates the clustering process due to its scale.

---

### **Overall Structure**
The code is organized into the following parts:

#### 1. **`Point` Class**
   - Represents a single data point in n-dimensional space.
   - **Attributes**:
     - `features`: A vector of doubles representing the point's coordinates.
     - `cluster_id`: The ID of the cluster the point belongs to.
   - **Methods**:
     - `getDimension()`: Returns the number of features (dimensions) of the point.
     - `getFeature(index)`: Returns the value of a specific feature.
     - `getFeatures()`: Returns all features as a vector.
     - `setCluster(id)`: Assigns the point to a cluster.
     - `getCluster()`: Returns the cluster ID.
     - `distance(other)`: Computes the Euclidean distance to another point.
     - `print()`: Displays the point's features and cluster ID.

#### 2. **`KMeans` Class**
   - Implements the K-Means clustering algorithm.
   - **Attributes**:
     - `k`: The number of clusters.
     - `max_iterations`: The maximum number of iterations for convergence.
     - `centroids`: A vector of `Point` objects representing the cluster centers.
   - **Methods**:
     - `initializeCentroids(data)`: Randomly selects `k` points from the dataset as initial centroids.
     - Other methods (not fully shown in the code) would include:
       - `assignClusters(data)`: Assigns each point to the nearest centroid.
       - `updateCentroids(data)`: Recalculates centroids based on current cluster assignments.
       - `train(data)`: Runs the K-Means algorithm until convergence.

#### 3. **Main Function**
   - **Steps**:
     1. Generates synthetic customer data.
     2. Normalizes the data.
     3. Initializes the K-Means model with `k=3` clusters.
     4. Trains the model.
     5. Displays the results.

---

### **How the Parts Work Together**
1. The `Point` class provides the foundation for representing and manipulating data points.
2. The `KMeans` class uses the `Point` class to:
   - Compute distances between points.
   - Assign points to clusters.
   - Update centroids.
3. The `main` function orchestrates the process:
   - Generates or loads data.
   - Normalizes the data.
   - Runs the K-Means algorithm.
   - Outputs the results.

---

### **Problem Being Solved**
The code solves the problem of **grouping similar data points into clusters**. For example:
- In customer segmentation, it could group customers based on income and spending habits.
- In image processing, it could group pixels with similar colors.

---

### **Approach Taken**
1. **Data Representation**:
   - Each data point is represented as a `Point` object with n-dimensional features.
2. **Clustering**:
   - The K-Means algorithm iteratively assigns points to clusters and updates centroids.
3. **Normalization**:
   - Ensures that all features contribute equally to the clustering process.

---

### **Summary**
This code is a well-structured implementation of the K-Means clustering algorithm. It uses object-oriented programming principles to encapsulate data points and clustering logic, making it modular and reusable. The code is designed to handle n-dimensional data, making it versatile for various applications.

Let me know if you'd like a line-by-line explanation or suggestions for improvements!