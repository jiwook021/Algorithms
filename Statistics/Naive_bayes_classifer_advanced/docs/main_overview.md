# Code Overview: main.cpp

This C++ code implements a **Naive Bayes classifier**, a fundamental machine learning algorithm used for classification tasks. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code is designed to implement a **Naive Bayes classifier**, which is a probabilistic machine learning model used to classify data points into categories (classes) based on their features. Specifically:
1. It works with **2D data points** (each data point has two features: `x1` and `x2`).
2. It uses **Gaussian (Normal) distributions** to model the probability distribution of each feature for each class.
3. It can classify new data points by calculating the probability that they belong to each class and selecting the most likely class.

The Naive Bayes classifier is based on **Bayes' Theorem**, which calculates the probability of a class given the observed features. The "naive" assumption is that the features are conditionally independent given the class, which simplifies the calculations.

---

### **Main Functionality**
1. **Data Representation**:
   - The `DataPoint` class represents a single data point with two features (`x1`, `x2`) and an optional class label (`label_`).
   - The label is binary (0 or 1), and the class provides methods to get/set the label and validate it.

2. **Probability Distributions**:
   - The `Distribution` abstract base class defines an interface for probability distributions.
   - The `GaussianDistribution` class implements a Gaussian (Normal) distribution, which is used to model the probability density of each feature for each class.

3. **Feature Statistics**:
   - The `FeatureStats` class (partially shown) calculates and stores statistics (mean, variance, etc.) for a feature across all data points of the same class. These statistics are used to estimate the parameters of the Gaussian distributions.

4. **Naive Bayes Classifier**:
   - The classifier (not fully shown in the provided code) would use the `DataPoint`, `GaussianDistribution`, and `FeatureStats` classes to:
     - Train the model by estimating the mean and variance of each feature for each class.
     - Classify new data points by calculating the posterior probability for each class and selecting the most likely one.

---

### **Algorithms Used**
1. **Gaussian Probability Density**:
   - The `GaussianDistribution` class implements the probability density function (PDF) for a Gaussian distribution:
     \[
     P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
     \]
   - The code pre-calculates the normalization factor for efficiency and guards against numerical underflow.

2. **Welford's Online Algorithm**:
   - The `FeatureStats` class uses Welford's algorithm to update the mean and variance incrementally as new data points are added. This is more numerically stable than calculating these statistics in a single pass.

3. **Naive Bayes Classification**:
   - The classifier would use Bayes' Theorem to calculate the posterior probability of each class given the features:
     \[
     P(\text{class} \mid x_1, x_2) \propto P(\text{class}) \cdot P(x_1 \mid \text{class}) \cdot P(x_2 \mid \text{class})
     \]
   - The "naive" assumption is that \( P(x_1, x_2 \mid \text{class}) = P(x_1 \mid \text{class}) \cdot P(x_2 \mid \text{class}) \).

---

### **Overall Structure**
The code is organized into several classes, each with a specific responsibility:
1. **`DataPoint`**:
   - Represents a single data point with features and an optional label.
   - Provides methods to get/set the label and validate it.
   - Includes a static method to create a `DataPoint` from user input.

2. **`Distribution`**:
   - Abstract base class for probability distributions.
   - Defines methods for calculating probability density and log probability density.

3. **`GaussianDistribution`**:
   - Implements a Gaussian distribution with mean and variance.
   - Provides methods to calculate the probability density and log probability density.

4. **`FeatureStats`**:
   - Encapsulates statistics (mean, variance, etc.) for a feature across data points of the same class.
   - Uses Welford's algorithm to update statistics incrementally.

---

### **How the Parts Work Together**
1. **Training**:
   - The classifier would use the `FeatureStats` class to calculate the mean and variance of each feature for each class.
   - These statistics are used to create `GaussianDistribution` objects for each feature and class.

2. **Classification**:
   - For a new `DataPoint`, the classifier would:
     - Calculate the probability density of each feature using the corresponding `GaussianDistribution`.
     - Use Bayes' Theorem to calculate the posterior probability for each class.
     - Assign the class with the highest probability.

3. **User Interaction**:
   - The `DataPoint::fromUserInput` method allows users to input new data points for classification.

---

### **Problem Being Solved**
The code solves the problem of **binary classification**:
- Given a set of labeled 2D data points, the classifier learns the probability distribution of each feature for each class.
- It can then classify new, unlabeled data points into one of the two classes.

---

### **Approach Taken**
1. **Object-Oriented Design**:
   - The code uses classes to encapsulate data and behavior, making it modular and reusable.
   - For example, the `GaussianDistribution` class can be reused in other contexts that require Gaussian probability calculations.

2. **Numerical Stability**:
   - The code includes safeguards against numerical issues, such as underflow in the Gaussian PDF and the use of Welford's algorithm for stable variance calculation.

3. **Error Handling**:
   - The code includes checks for invalid input (e.g., non-positive variance, invalid labels) and throws exceptions with descriptive messages.

---

### **Summary**
This code implements a Naive Bayes classifier for binary classification of 2D data points. It uses Gaussian distributions to model feature probabilities and is designed with modularity, numerical stability, and error handling in mind. The classifier can be trained on labeled data and used to classify new data points based on their features.