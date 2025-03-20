# Code Overview: main.cpp

This C++ code is a partial implementation of a **machine learning system** that performs **binary classification** using a **Support Vector Machine (SVM)**. The code focuses on **data preprocessing** and **kernel selection**, which are critical steps in building an SVM model. Let’s break down the purpose, functionality, and structure of the code in detail:

---

### **Purpose of the Code**
The code aims to:
1. **Preprocess data** for machine learning by normalizing features using **Z-score normalization** (also called standardization).
2. Allow the user to select a **kernel function** for the SVM, which determines how the algorithm will separate the data into two classes.
3. Prepare the data for training an SVM model by ensuring the features are on the same scale and ready for kernel-based transformations.

The problem being solved is **binary classification**, where the goal is to separate data points into two classes (labeled `-1` and `1`) based on their features (`x1` and `x2`).

---

### **Main Functionality**
1. **Data Representation**:
   - The `DataPoint` struct represents a single data point with two features (`x1`, `x2`) and a binary label (`label`).
   - This structure is used to store the dataset and pass it through the preprocessing pipeline.

2. **Data Preprocessing**:
   - The `DataPreprocessor` class performs **Z-score normalization** on the dataset. This involves:
     - Calculating the **mean** and **standard deviation** of each feature (`x1` and `x2`).
     - Scaling the features so that they have a mean of `0` and a standard deviation of `1`.
   - Normalization is crucial for machine learning algorithms like SVM because it ensures that all features contribute equally to the model's decision-making process.

3. **Kernel Selection**:
   - The code prompts the user to select a kernel type for the SVM:
     - **Linear kernel**: Suitable for linearly separable data.
     - **RBF (Gaussian) kernel**: Suitable for non-linearly separable data.
   - The kernel choice determines how the SVM will transform the data to find the optimal decision boundary.

4. **Error Handling**:
   - The code includes robust error handling to ensure the preprocessor is fitted before transforming data and to handle invalid inputs (e.g., empty datasets).

---

### **Algorithms and Techniques Used**
1. **Z-score Normalization**:
   - Formula: `normalized_value = (value - mean) / standard_deviation`
   - This ensures that the features have a mean of `0` and a standard deviation of `1`, making them comparable and improving the performance of the SVM.

2. **Support Vector Machine (SVM)**:
   - Although the SVM implementation is not fully shown in the code, the preprocessing and kernel selection are essential steps for SVM training.
   - The kernel function determines how the SVM will map the data into a higher-dimensional space to find a separating hyperplane.

3. **Object-Oriented Programming (OOP)**:
   - The code uses classes (`DataPreprocessor`) and structs (`DataPoint`) to encapsulate related functionality and data, making the code modular and reusable.

---

### **Overall Structure**
The code is organized into the following components:
1. **DataPoint Struct**:
   - Represents a single data point with features and a label.
   - Includes a constructor for easy initialization.

2. **DataPreprocessor Class**:
   - Encapsulates the logic for normalizing data.
   - Provides methods to:
     - `fit()`: Calculate the mean and standard deviation of the dataset.
     - `transform()`: Normalize the dataset using the calculated parameters.
     - `transform_point()`: Normalize a single data point.
     - `print_parameters()`: Display the mean and standard deviation used for normalization.

3. **Main Function**:
   - Creates a sample dataset.
   - Preprocesses the data using the `DataPreprocessor`.
   - Prompts the user to select a kernel type for the SVM.
   - (Incomplete) Initializes the selected kernel for the SVM.

---

### **How the Parts Work Together**
1. The `DataPoint` struct is used to store the dataset, which is passed to the `DataPreprocessor`.
2. The `DataPreprocessor` calculates the mean and standard deviation of the dataset during the `fit()` step.
3. The `transform()` method normalizes the dataset using the calculated parameters.
4. The user selects a kernel type, which determines how the SVM will process the normalized data.
5. The preprocessed data and selected kernel are then ready to be used for training the SVM (though the SVM implementation is not fully shown in the code).

---

### **Problem Being Solved**
The code addresses the problem of **binary classification**, where the goal is to separate data points into two classes based on their features. The preprocessing step ensures that the features are properly scaled, which is critical for the SVM to perform well. The kernel selection step allows the SVM to handle both linear and non-linear decision boundaries.

---

### **Approach Taken**
1. **Modular Design**:
   - The code is divided into clear components (`DataPoint`, `DataPreprocessor`, and the main function), making it easy to understand, extend, and debug.

2. **Error Handling**:
   - The code checks for invalid inputs (e.g., empty datasets) and ensures the preprocessor is fitted before transforming data.

3. **User Interaction**:
   - The user is prompted to select a kernel type, making the system interactive and flexible.

4. **Scalability**:
   - The `DataPreprocessor` class is designed to handle datasets of any size, and the normalization logic is efficient.

---

### **Summary**
This code is a well-structured implementation of data preprocessing and kernel selection for an SVM-based binary classification system. It demonstrates key concepts in machine learning, such as feature scaling, modular design, and user interaction. While the SVM implementation is incomplete, the provided code lays a strong foundation for building a robust classification system.