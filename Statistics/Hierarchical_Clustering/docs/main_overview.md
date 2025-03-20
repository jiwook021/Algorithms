# Code Overview: main.cpp

This C++ code implements a **hierarchical clustering algorithm**, specifically using the **single-linkage** method. Hierarchical clustering is a type of unsupervised machine learning algorithm used to group similar data points into clusters. The goal is to organize the data into a hierarchy of clusters, where clusters at one level are merged into larger clusters at the next level, until all data points are in a single cluster.

Let’s break down the purpose, functionality, and structure of the code step by step:

---

### **1. Problem Being Solved**
The code solves the problem of **grouping a set of 2D points into clusters** based on their similarity (proximity). The algorithm starts by treating each point as its own cluster and then iteratively merges the closest clusters until only one cluster remains. This process creates a hierarchy of clusters, which can be visualized as a dendrogram (a tree-like diagram).

---

### **2. Main Functionality**
The code performs the following steps:
1. **Initialization**: Each data point is treated as its own cluster.
2. **Distance Calculation**: The Euclidean distance between every pair of clusters is calculated.
3. **Cluster Merging**: The two closest clusters are merged into a single cluster.
4. **Iteration**: Steps 2 and 3 are repeated until only one cluster remains.
5. **Output**: The algorithm prints the merge steps and the remaining number of clusters after each merge.

---

### **3. Algorithms Used**
The code uses the **single-linkage hierarchical clustering algorithm**, which is a type of agglomerative clustering. Agglomerative clustering is a bottom-up approach where each data point starts as its own cluster, and clusters are merged based on their proximity. The single-linkage method defines the distance between two clusters as the **minimum distance between any pair of points from the two clusters**.

---

### **4. Overall Structure**
The code is organized into several components:
- **Data Structures**:
  - `Point`: Represents a 2D point with `x` and `y` coordinates.
  - `Cluster`: Represents a group of points (a cluster).
- **Functions**:
  - `distance`: Computes the Euclidean distance between two points.
  - `cluster_distance`: Computes the minimum distance between two clusters (single-linkage).
  - `find_closest_clusters`: Finds the pair of clusters with the smallest distance.
  - `merge_clusters`: Merges two clusters into one.
  - `hierarchical_clustering`: Implements the hierarchical clustering algorithm.
- **Main Function**:
  - Initializes a dataset of 2D points and runs the clustering algorithm.

---

### **5. How the Code Works Together**
1. **Initialization**:
   - The `main` function initializes a dataset of 2D points.
   - The `hierarchical_clustering` function starts by creating a cluster for each point.

2. **Distance Calculation**:
   - The `distance` function calculates the Euclidean distance between two points.
   - The `cluster_distance` function uses `distance` to find the minimum distance between all pairs of points in two clusters.

3. **Finding Closest Clusters**:
   - The `find_closest_clusters` function iterates through all pairs of clusters and identifies the pair with the smallest distance using `cluster_distance`.

4. **Merging Clusters**:
   - The `merge_clusters` function combines the points of two clusters into a single cluster.
   - The merged cluster replaces the two original clusters in the list of clusters.

5. **Iteration**:
   - The algorithm repeats the process of finding the closest clusters and merging them until only one cluster remains.

6. **Output**:
   - After each merge, the algorithm prints the points in the merged cluster and the number of remaining clusters.

---

### **6. Example Walkthrough**
For the dataset:
```
{1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
```
The algorithm might proceed as follows:
1. Start with 6 clusters (one for each point).
2. Merge the two closest points (e.g., `{1.0, 2.0}` and `{1.5, 1.8}`).
3. Repeat the process, merging clusters until all points are in a single cluster.

---

### **7. Key Concepts**
- **Euclidean Distance**: Measures the straight-line distance between two points in 2D space.
- **Single-Linkage**: Defines the distance between clusters as the minimum distance between any two points in the clusters.
- **Agglomerative Clustering**: A bottom-up approach where each point starts as its own cluster, and clusters are merged iteratively.

---

### **8. Why This Code is Useful**
This code is useful for:
- Grouping similar data points into clusters.
- Understanding the hierarchical relationships between data points.
- Applications in data analysis, pattern recognition, and machine learning.

---

### **9. Limitations**
- The algorithm has a time complexity of **O(n³)** in the worst case, making it inefficient for large datasets.
- It uses a hardcoded dataset, which limits its flexibility.
- The single-linkage method can lead to "chaining," where clusters are merged based on a single close pair of points, even if the rest of the points are far apart.

---

### **10. Summary**
This code implements a hierarchical clustering algorithm using the single-linkage method to group 2D points into clusters. It starts with each point as its own cluster, iteratively merges the closest clusters, and outputs the merge steps until only one cluster remains. The code is well-structured and demonstrates the core concepts of hierarchical clustering, making it a great educational example.

Let me know if you'd like to dive deeper into any specific part of the code or discuss potential improvements!