# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, even if you’re a beginner.

---

### **1. The `Point` Structure**
```cpp
struct Point {
    double x;  // Feature 1
    double y;  // Feature 2
};
```

#### **What it does:**
- This defines a **structure** (a custom data type) called `Point`.
- A `Point` represents a 2D point with two properties: `x` and `y`, which are both `double` (decimal) values.

#### **Why it’s used:**
- In 2D data analysis, each data point has two features (e.g., height and weight, or x and y coordinates). The `Point` structure is a convenient way to store and manipulate these pairs of values.

#### **Example:**
- If you have a point `{3.0, 4.0}`, it means the point is located at `x = 3.0` and `y = 4.0` on a 2D plane.

---

### **2. The `PCA` Class**
The `PCA` class encapsulates all the logic for performing Principal Component Analysis. Let’s break it down step by step.

---

#### **2.1 Private Members**
```cpp
private:
    Point mean;           // Mean of the dataset
    double eigenvector_x; // x-component of the principal eigenvector
    double eigenvector_y; // y-component of the principal eigenvector
```

#### **What they do:**
- `mean`: Stores the average (mean) of all the points in the dataset.
- `eigenvector_x` and `eigenvector_y`: Store the direction of the **principal component** (the direction of maximum variance in the data).

#### **Why they’re used:**
- The mean is needed to **center the data** (subtract the mean from each point), which is a crucial step in PCA.
- The eigenvector represents the direction in which the data varies the most. Projecting data onto this direction reduces its dimensionality while preserving as much variance as possible.

---

#### **2.2 `compute_mean` Function**
```cpp
Point compute_mean(const std::vector<Point>& data) const {
    Point m = {0.0, 0.0};
    for (const auto& p : data) {
        m.x += p.x;
        m.y += p.y;
    }
    m.x /= data.size();
    m.y /= data.size();
    return m;
}
```

#### **What it does:**
- Computes the **mean** (average) of all the points in the dataset.

#### **Step-by-step breakdown:**
1. Initialize a `Point` called `m` with `x = 0.0` and `y = 0.0`.
2. Loop through each point `p` in the dataset:
   - Add `p.x` to `m.x`.
   - Add `p.y` to `m.y`.
3. After the loop, divide `m.x` and `m.y` by the number of points (`data.size()`) to get the average.
4. Return the computed mean.

#### **Why it’s used:**
- The mean is subtracted from each point to center the data around the origin. This ensures that the PCA analysis focuses on the **variance** of the data rather than its absolute position.

#### **Example:**
- If the dataset is `{ {1, 2}, {2, 3}, {3, 4} }`, the mean is:
  - `m.x = (1 + 2 + 3) / 3 = 2.0`
  - `m.y = (2 + 3 + 4) / 3 = 3.0`

---

#### **2.3 `compute_covariance` Function**
```cpp
std::vector<std::vector<double>> compute_covariance(const std::vector<Point>& data) const {
    std::vector<std::vector<double>> cov(2, std::vector<double>(2, 0.0));
    for (const auto& p : data) {
        double dx = p.x - mean.x;
        double dy = p.y - mean.y;
        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[1][0] += dy * dx;
        cov[1][1] += dy * dy;
    }
    cov[0][0] /= data.size();
    cov[0][1] /= data.size();
    cov[1][0] /= data.size();
    cov[1][1] /= data.size();
    return cov;
}
```

#### **What it does:**
- Computes the **covariance matrix** of the centered data.

#### **Step-by-step breakdown:**
1. Initialize a 2x2 matrix `cov` with all elements set to `0.0`.
2. Loop through each point `p` in the dataset:
   - Compute the centered values `dx = p.x - mean.x` and `dy = p.y - mean.y`.
   - Update the covariance matrix:
     - `cov[0][0] += dx * dx` (variance of x)
     - `cov[0][1] += dx * dy` (covariance of x and y)
     - `cov[1][0] += dy * dx` (covariance of y and x)
     - `cov[1][1] += dy * dy` (variance of y)
3. After the loop, divide each element of the matrix by the number of points to get the average.
4. Return the covariance matrix.

#### **Why it’s used:**
- The covariance matrix captures how the x and y coordinates vary together. It’s essential for identifying the principal component.

#### **Example:**
- For the dataset `{ {1, 2}, {2, 3}, {3, 4} }` and mean `{2.0, 3.0}`:
  - The centered points are `{ {-1, -1}, {0, 0}, {1, 1} }`.
  - The covariance matrix is:
    ```
    [ (1 + 0 + 1)/3,  (1 + 0 + 1)/3 ]
    [ (1 + 0 + 1)/3,  (1 + 0 + 1)/3 ]
    ```

---

#### **2.4 `compute_eigenvector` Function**
```cpp
void compute_eigenvector(const std::vector<std::vector<double>>& cov) {
    double a = cov[0][0], b = cov[0][1], c = cov[1][0], d = cov[1][1];
    double trace = a + d;
    double det = a * d - b * c;
    double discriminant = std::sqrt(trace * trace - 4 * det);
    double lambda1 = (trace + discriminant) / 2;  // Larger eigenvalue

    // Eigenvector for lambda1 (simplified for 2D)
    eigenvector_x = b;
    eigenvector_y = lambda1 - a;
    double norm = std::sqrt(eigenvector_x * eigenvector_x + eigenvector_y * eigenvector_y);
    eigenvector_x /= norm;
    eigenvector_y /= norm;
}
```

#### **What it does:**
- Computes the **principal eigenvector** (direction of maximum variance) from the covariance matrix.

#### **Step-by-step breakdown:**
1. Extract the elements of the covariance matrix:
   - `a = cov[0][0]`, `b = cov[0][1]`, `c = cov[1][0]`, `d = cov[1][1]`.
2. Compute the **trace** (sum of diagonal elements) and **determinant** (product of diagonal elements minus product of off-diagonal elements).
3. Compute the **discriminant** (used to solve the characteristic equation).
4. Compute the larger eigenvalue `lambda1` (represents the maximum variance).
5. Compute the eigenvector corresponding to `lambda1`:
   - `eigenvector_x = b`
   - `eigenvector_y = lambda1 - a`
6. Normalize the eigenvector to have a length of 1 (unit vector).

#### **Why it’s used:**
- The eigenvector represents the direction of maximum variance in the data. Projecting data onto this direction reduces dimensionality while preserving as much variance as possible.

#### **Example:**
- For the covariance matrix:
  ```
  [ 1, 1 ]
  [ 1, 1 ]
  ```
  - The eigenvalues are `2` and `0`.
  - The eigenvector for `lambda1 = 2` is `(1, 1)`, normalized to `(0.707, 0.707)`.

---

#### **2.5 `fit` Function**
```cpp
void fit(const std::vector<Point>& data) {
    // Step 1: Compute the mean
    mean = compute_mean(data);

    // Step 2: Compute the covariance matrix
    auto cov = compute_covariance(data);

    // Step 3: Compute the principal eigenvector
    compute_eigenvector(cov);
}
```

#### **What it does:**
- Trains the PCA model by computing the mean, covariance matrix, and principal eigenvector.

#### **Step-by-step breakdown:**
1. Compute the mean of the dataset using `compute_mean`.
2. Compute the covariance matrix using `compute_covariance`.
3. Compute the principal eigenvector using `compute_eigenvector`.

#### **Why it’s used:**
- This function encapsulates the training process of PCA, making it easy to use.

---

#### **2.6 `predict` Function**
```cpp
double predict(const Point& p) const {
    // Center the point
    double dx = p.x - mean.x;
    double dy = p.y - mean.y;
    // Dot product with the principal eigenvector
    return dx * eigenvector_x + dy * eigenvector_y;
}
```

#### **What it does:**
- Projects a new point onto the principal component.

#### **Step-by-step breakdown:**
1. Center the point by subtracting the mean.
2. Compute the dot product of the centered point and the principal eigenvector.

#### **Why it’s used:**
- The dot product measures how much of the point lies in the direction of the principal component, effectively reducing the 2D point to a 1D value.

---

#### **2.7 `print_direction` Function**
```cpp
void print_direction() const {
    std::cout << "Principal Component Direction: (" << eigenvector_x << ", " << eigenvector_y << ")\n";
}
```

#### **What it does:**
- Prints the direction of the principal component.

#### **Why it’s used:**
- Provides a way to visualize the direction of maximum variance.

---

### **3. The `main` Function**
```cpp
int main() {
    // Hardcoded dataset: 2D points with some correlation
    std::vector<Point> data = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
        {4.0, 5.0}, {5.0, 6.0}
    };

    // Initialize and train PCA
    PCA model;
    model.fit(data);

    // Show the direction of the principal component
    model.print_direction();

    // User input for a new point
    std::cout << "Enter x and y for a new point (e.g., 3.0 4.0): ";
    double x, y;
    std::cin >> x >> y;
    Point new_point = {x, y};

    // Project and display the result
    double projection = model.predict(new_point);
    std::cout << "Projected value for (" << x << ", " << y << "): " << projection << std::endl;

    return 0;
}
```

#### **What it does:**
- Demonstrates the PCA model by:
  1. Creating a dataset.
  2. Training the PCA model.
  3. Printing the principal component direction.
  4. Projecting a user-provided point onto the principal component.

#### **Step-by-step breakdown:**
1. Define a dataset of 2D points.
2. Create a `PCA` object and train it using `fit`.
3. Print the principal component direction using `print_direction`.
4. Prompt the user for a new point and project it using `predict`.
5. Display the projection result.

#### **Why it’s used:**
- This function ties everything together, demonstrating how to use the PCA class in practice.

---

### **Summary**
This code implements PCA for 2D data, identifying the direction of maximum variance and projecting new points onto this direction. It’s a great example of how to break down a complex algorithm into manageable steps and encapsulate it in a class for easy use.