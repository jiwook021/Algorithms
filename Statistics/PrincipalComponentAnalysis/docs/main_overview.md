# Code Overview: main.cpp

This C++ code implements **Principal Component Analysis (PCA)**, a fundamental technique in **dimensionality reduction** and **data analysis**. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The purpose of this code is to perform **PCA on a 2D dataset** to identify the **principal component** (the direction of maximum variance in the data). Once the principal component is determined, the code can project any new 2D point onto this direction, effectively reducing the 2D data to a 1D representation. This is useful for simplifying data, visualizing trends, or preparing data for further analysis.

---

### **Main Functionality**
1. **PCA Training**:
   - The code computes the **mean** of the dataset.
   - It calculates the **covariance matrix** of the centered data.
   - It determines the **principal eigenvector** (the direction of maximum variance) from the covariance matrix.

2. **Projection**:
   - Given a new 2D point, the code projects it onto the principal component, reducing it to a single value.

3. **Visualization**:
   - The code outputs the direction of the principal component and the projection of a user-provided point.

---

### **Algorithms Used**
1. **Mean Calculation**:
   - The mean of the dataset is computed by summing all the x and y coordinates and dividing by the number of points.

2. **Covariance Matrix Calculation**:
   - The covariance matrix is computed to measure how the x and y coordinates vary together. For 2D data, this is a 2x2 matrix.

3. **Eigenvector Computation**:
   - The principal eigenvector (the direction of maximum variance) is computed using a simplified method for 2x2 matrices. This involves solving the characteristic equation of the covariance matrix.

4. **Projection**:
   - A new point is projected onto the principal component by taking the dot product of the centered point and the principal eigenvector.

---

### **Overall Structure**
The code is organized into two main components:
1. **`Point` Structure**:
   - Represents a 2D point with `x` and `y` coordinates.

2. **`PCA` Class**:
   - Contains private methods for computing the mean, covariance matrix, and eigenvector.
   - Provides public methods for training the PCA model (`fit`), projecting new points (`predict`), and displaying the principal component direction (`print_direction`).

3. **`main` Function**:
   - Initializes a dataset of 2D points.
   - Trains the PCA model on the dataset.
   - Prompts the user for a new point and projects it onto the principal component.

---

### **How the Parts Work Together**
1. **Dataset Preparation**:
   - The dataset is hardcoded as a vector of `Point` objects.

2. **PCA Training**:
   - The `fit` method computes the mean, covariance matrix, and principal eigenvector in sequence.

3. **User Interaction**:
   - The user provides a new point, which is projected onto the principal component using the `predict` method.

4. **Output**:
   - The direction of the principal component and the projection of the new point are displayed.

---

### **Problem Being Solved**
The code solves the problem of **identifying the most important direction (principal component)** in a 2D dataset and **reducing the dimensionality** of the data by projecting it onto this direction. This is useful for:
- Simplifying data for visualization.
- Reducing noise by focusing on the most significant variation.
- Preparing data for machine learning algorithms that perform better with lower-dimensional data.

---

### **Approach Taken**
1. **Mean-Centering**:
   - The data is centered by subtracting the mean, which is a standard step in PCA to ensure the analysis focuses on variance rather than the absolute position of the data.

2. **Covariance Matrix**:
   - The covariance matrix captures the relationships between the x and y coordinates, which is essential for identifying the principal component.

3. **Eigenvector Computation**:
   - The principal eigenvector is computed using a simplified method for 2x2 matrices, which is efficient and sufficient for this use case.

4. **Projection**:
   - The projection of a new point onto the principal component is computed using the dot product, which measures how much of the point lies in the direction of the principal component.

---

### **Key Takeaways**
- The code is a **simplified implementation of PCA** tailored for 2D data.
- It demonstrates the core steps of PCA: mean-centering, covariance computation, eigenvector calculation, and projection.
- The code is modular, with clear separation of concerns between data representation (`Point`), PCA logic (`PCA` class), and user interaction (`main` function).

This implementation is a great starting point for understanding PCA and can be extended to handle higher-dimensional data or more complex scenarios.