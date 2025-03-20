# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Gaussian Probability Density Function (PDF)**
```cpp
double gaussian_pdf(double x, double mean, double variance) {
    const double PI = 3.141592653589793;
    double stddev = std::sqrt(variance);
    return (1.0 / (stddev * std::sqrt(2.0 * PI))) * std::exp(-std::pow(x - mean, 2) / (2.0 * variance));
}
```

#### **What It Does**
This function calculates the probability of a value `x` under a Gaussian (normal) distribution with a given `mean` and `variance`.

#### **Breakdown**
1. **Inputs:**
   - `x`: The data point for which we want to calculate the probability.
   - `mean`: The center of the Gaussian distribution.
   - `variance`: A measure of how spread out the distribution is.

2. **Constants and Calculations:**
   - `PI`: A constant representing the mathematical value π (3.14159...).
   - `stddev`: The standard deviation, which is the square root of the variance. It measures the spread of the distribution.

3. **Formula:**
   The Gaussian PDF formula is:
   \[
   f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
   \]
   - The first part, `1.0 / (stddev * std::sqrt(2.0 * PI))`, normalizes the distribution so that the total area under the curve is 1.
   - The second part, `std::exp(-std::pow(x - mean, 2) / (2.0 * variance))`, calculates the exponential term that determines the shape of the distribution.

4. **Why It’s Used:**
   - The Gaussian PDF is used to compute the likelihood of a data point belonging to a particular cluster. This is essential for the E-step in the EM algorithm.

---

### **2. GaussianComponent Struct**
```cpp
struct GaussianComponent {
    double mean;
    double variance;
    double weight;
};
```

#### **What It Does**
This struct represents a single Gaussian component in the mixture model. It stores the parameters of the Gaussian distribution:
- `mean`: The center of the distribution.
- `variance`: The spread of the distribution.
- `weight`: The importance of this component in the mixture.

#### **Why It’s Used:**
- The GMM assumes that the data is generated from a mixture of Gaussian distributions. Each component represents one of these distributions.

---

### **3. GMM Class**
The `GMM` class is the core of the implementation. Let’s break it down step by step.

#### **Private Members**
```cpp
private:
    GaussianComponent comp1, comp2;          // Two Gaussian components
    std::vector<double> responsibilities1;   // P(cluster1 | x_i)
    std::vector<double> responsibilities2;   // P(cluster2 | x_i)
    double tolerance = 1e-4;                 // Convergence threshold
    int max_iterations = 100;                // Maximum EM iterations
```

1. **Gaussian Components (`comp1`, `comp2`):**
   - These represent the two Gaussian distributions in the mixture model.

2. **Responsibilities (`responsibilities1`, `responsibilities2`):**
   - These vectors store the probabilities that each data point belongs to `comp1` or `comp2`.

3. **Tolerance and Max Iterations:**
   - `tolerance`: A small value used to check if the algorithm has converged (i.e., if the parameters have stopped changing significantly).
   - `max_iterations`: The maximum number of times the EM algorithm will run.

#### **Constructor**
```cpp
GMM() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    comp1 = {static_cast<double>(std::rand() % 100), 1.0, 0.5};
    comp2 = {static_cast<double>(std::rand() % 100), 1.0, 0.5};
}
```

1. **Random Initialization:**
   - The constructor initializes the parameters of `comp1` and `comp2` with random values.
   - `std::srand` seeds the random number generator with the current time to ensure different results each time the program runs.
   - `std::rand() % 100` generates a random integer between 0 and 99, which is cast to a double for the mean.

2. **Why Random Initialization?**
   - The EM algorithm is sensitive to initial conditions. Random initialization helps avoid getting stuck in poor local optima.

---

### **4. E-Step**
```cpp
void e_step(const std::vector<double>& data) {
    responsibilities1.clear();
    responsibilities2.clear();
    for (double x : data) {
        double pdf1 = gaussian_pdf(x, comp1.mean, comp1.variance) * comp1.weight;
        double pdf2 = gaussian_pdf(x, comp2.mean, comp2.variance) * comp2.weight;
        double total = pdf1 + pdf2;
        responsibilities1.push_back(pdf1 / total);
        responsibilities2.push_back(pdf2 / total);
    }
}
```

#### **What It Does**
The E-step computes the responsibilities, which are the probabilities that each data point belongs to each cluster.

#### **Breakdown**
1. **Clear Responsibilities:**
   - The `responsibilities1` and `responsibilities2` vectors are cleared to prepare for new calculations.

2. **Loop Through Data:**
   - For each data point `x`, the code calculates:
     - `pdf1`: The probability of `x` under `comp1`, weighted by `comp1.weight`.
     - `pdf2`: The probability of `x` under `comp2`, weighted by `comp2.weight`.

3. **Normalize Probabilities:**
   - The total probability (`total`) is the sum of `pdf1` and `pdf2`.
   - The responsibilities are normalized by dividing `pdf1` and `pdf2` by `total`.

4. **Why Normalize?**
   - Normalization ensures that the responsibilities sum to 1, making them valid probabilities.

---

### **5. M-Step**
```cpp
void m_step(const std::vector<double>& data) {
    double sum_resp1 = 0.0, sum_resp2 = 0.0;
    double weighted_sum1 = 0.0, weighted_sum2 = 0.0;
    double weighted_var1 = 0.0, weighted_var2 = 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        sum_resp1 += responsibilities1[i];
        sum_resp2 += responsibilities2[i];
        weighted_sum1 += responsibilities1[i] * data[i];
        weighted_sum2 += responsibilities2[i] * data[i];
        weighted_var1 += responsibilities1[i] * std::pow(data[i] - comp1.mean, 2);
        weighted_var2 += responsibilities2[i] * std::pow(data[i] - comp2.mean, 2);
    }

    // Update means
    comp1.mean = weighted_sum1 / sum_resp1;
    comp2.mean = weighted_sum2 / sum_resp2;

    // Update variances
    comp1.variance = weighted_var1 / sum_resp1;
    comp2.variance = weighted_var2 / sum_resp2;

    // Update weights
    comp1.weight = sum_resp1 / data.size();
    comp2.weight = sum_resp2 / data.size();
}
```

#### **What It Does**
The M-step updates the parameters of the Gaussian components based on the responsibilities computed in the E-step.

#### **Breakdown**
1. **Initialize Sums:**
   - `sum_resp1` and `sum_resp2`: Sum of responsibilities for each cluster.
   - `weighted_sum1` and `weighted_sum2`: Weighted sums of the data points for each cluster.
   - `weighted_var1` and `weighted_var2`: Weighted sums of squared differences for each cluster.

2. **Loop Through Data:**
   - For each data point, the code updates the sums using the responsibilities.

3. **Update Parameters:**
   - **Means:** The new mean is the weighted average of the data points.
   - **Variances:** The new variance is the weighted average of the squared differences from the mean.
   - **Weights:** The new weight is the average responsibility for the cluster.

4. **Why Update Parameters?**
   - The M-step maximizes the likelihood of the data given the current responsibilities.

---

### **6. Fit Method**
```cpp
void fit(const std::vector<double>& data) {
    for (int iter = 0; iter < max_iterations; ++iter) {
        double prev_mean1 = comp1.mean;
        double prev_mean2 = comp2.mean;

        e_step(data);
        m_step(data);

        // Check for convergence
        if (std::abs(comp1.mean - prev_mean1) < tolerance && std::abs(comp2.mean - prev_mean2) < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
    }
}
```

#### **What It Does**
The `fit` method runs the EM algorithm until convergence or the maximum number of iterations is reached.

#### **Breakdown**
1. **Loop Through Iterations:**
   - The code runs the E-step and M-step repeatedly.

2. **Check for Convergence:**
   - After each iteration, the code checks if the means of the Gaussian components have stopped changing significantly (i.e., the change is less than `tolerance`).

3. **Why Check Convergence?**
   - Convergence indicates that the parameters have stabilized, and further iterations are unlikely to improve the model.

---

### **7. Predict Method**
```cpp
int predict(double x) const {
    double prob1 = gaussian_pdf(x, comp1.mean, comp1.variance) * comp1.weight;
    double prob2 = gaussian_pdf(x, comp2.mean, comp2.variance) * comp2.weight;
    return (prob1 > prob2) ? 0 : 1;
}
```

#### **What It Does**
The `predict` method assigns a data point to the cluster with the highest probability.

#### **Breakdown**
1. **Calculate Probabilities:**
   - The code calculates the probability of `x` under each Gaussian component, weighted by the component’s weight.

2. **Assign Cluster:**
   - The data point is assigned to the cluster with the higher probability.

---

### **8. Main Function**
```cpp
int main() {
    std::vector<double> data = {1.0, 2.0, 1.5, 2.5, 10.0, 11.0, 9.5, 10.5, 12.0};

    GMM model;
    model.fit(data);

    model.print_parameters();

    std::cout << "\nCluster assignments:\n";
    for (double x : data) {
        int cluster = model.predict(x);
        std::cout << "Value " << x << " -> Cluster " << cluster +1 << "\n";
    }

    return 0;
}
```

#### **What It Does**
The `main` function demonstrates the GMM by fitting it to a dataset and displaying the results.

#### **Breakdown**
1. **Dataset:**
   - The dataset contains two distinct groups of values.

2. **Fit the Model:**
   - The GMM is fitted to the data using the `fit` method.

3. **Print Parameters:**
   - The learned parameters (means, variances, and weights) are printed.

4. **Cluster Assignments:**
   - Each data point is assigned to a cluster, and the results are displayed.

---

### **Summary**
This code implements a **Gaussian Mixture Model** using the **EM algorithm** to cluster data into two groups. It demonstrates key concepts in unsupervised learning, including probabilistic modeling, iterative optimization, and parameter estimation. The modular design and clear structure make it easy to understand and extend.

Let me know if you’d like to dive deeper into any specific part!