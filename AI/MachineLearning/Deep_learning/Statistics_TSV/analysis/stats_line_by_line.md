# Step-by-Step Explanation: stats.cpp

Let's dive into the `stats.cpp` code file, breaking it down step-by-step to ensure a comprehensive understanding of each part. We'll approach this as if you're learning to program, explaining each concept thoroughly.

### Header Files and Namespace

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
```

1. **Header Files**:
   - `#include <iostream>`: This includes the input-output stream library, which is typically used for reading from and writing to the console. However, in this code, it's not directly used.
   - `#include <vector>`: This includes the vector library, which provides the `vector` data structure. A vector is a dynamic array that can change size, allowing us to store a list of elements.
   - `#include <algorithm>`: This includes algorithms like `sort`, which are used to perform operations on data structures.
   - `#include <cmath>`: This includes mathematical functions like `sqrt` (square root) and `modf` (modulus function for floating-point numbers).

2. **Namespace**:
   - `using namespace std;`: This line allows us to use all the standard library functions and objects without prefixing them with `std::`. For example, we can write `vector` instead of `std::vector`.

### Function: `count`

```cpp
int count(vector<double> v) {
    int count_number;
    count_number = v.size();
    return (count_number);
}
```

1. **Purpose**: This function calculates the number of elements in the vector `v`.

2. **Logic**:
   - `v.size()`: This method returns the number of elements in the vector `v`.
   - `int count_number;`: Declares an integer variable `count_number`.
   - `count_number = v.size();`: Assigns the size of the vector to `count_number`.
   - `return (count_number);`: Returns the count of elements.

3. **Example**:
   - If `v = {1.0, 2.0, 3.0}`, `v.size()` returns `3`, so `count_number` is `3`.

### Function: `sum`

```cpp
double sum(vector<double> v) {
    double sum_number = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum_number = sum_number + v.at(i);
    }
    return (sum_number);
}
```

1. **Purpose**: This function calculates the sum of all elements in the vector `v`.

2. **Logic**:
   - `double sum_number = 0;`: Initializes a variable `sum_number` to store the sum, starting at `0`.
   - `for (size_t i = 0; i < v.size(); ++i)`: A loop that iterates over each element in the vector. `size_t` is an unsigned integer type used for sizes.
     - `v.at(i)`: Accesses the element at index `i`.
     - `sum_number = sum_number + v.at(i);`: Adds the current element to `sum_number`.
   - `return (sum_number);`: Returns the total sum.

3. **Example**:
   - If `v = {1.0, 2.0, 3.0}`, the loop adds `1.0 + 2.0 + 3.0`, resulting in `sum_number = 6.0`.

### Function: `mean`

```cpp
double mean(vector<double> v) {
    int sum_number = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum_number = sum_number + v.at(i);
    }
    int mean_number = sum_number / v.size();
    return (mean_number);
}
```

1. **Purpose**: This function calculates the mean (average) of the elements in the vector `v`.

2. **Logic**:
   - `int sum_number = 0;`: Initializes `sum_number` to store the sum.
   - The loop calculates the sum as in the `sum` function.
   - `int mean_number = sum_number / v.size();`: Divides the sum by the number of elements to get the mean.
   - `return (mean_number);`: Returns the mean.

3. **Issues**:
   - **Data Type**: The function uses `int` for `sum_number` and `mean_number`, which can lead to incorrect results due to integer division. It should use `double` to handle decimal values properly.

4. **Example**:
   - If `v = {1.0, 2.0, 3.0}`, the sum is `6.0`, and the mean is `6.0 / 3 = 2.0`.

### Function: `median`

```cpp
double median(vector<double> v) {
    std::sort(v.begin(), v.end());
    double median_number;
    if (v.size() % 2 == 0) {
        median_number = v.at((v.size() / 2) + 0.5) - v.at((v.size() / 2) - 0.5);
        return (median_number);
    }
    else {
        median_number = v.at((v.size() / 2) + 0.5);
        return (median_number);
    }
}
```

1. **Purpose**: This function calculates the median, which is the middle value of the sorted dataset.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector `v` in ascending order.
   - `if (v.size() % 2 == 0)`: Checks if the number of elements is even.
     - **Even Case**: The median is the average of the two middle numbers.
     - **Odd Case**: The median is the middle number.
   - **Issues**: The calculation of the median is incorrect due to the use of `+ 0.5` and `- 0.5`. It should correctly average the two middle elements for even-sized vectors.

3. **Example**:
   - If `v = {3.0, 1.0, 2.0}`, after sorting, `v = {1.0, 2.0, 3.0}`. The median is `2.0`.

### Function: `mode`

```cpp
double mode(vector<double> v) {
    std::sort(v.begin(), v.end());
    int count_max = 1;
    double mode_final = v.at(0);
    for (size_t i = 0; i < v.size(); ++i) {
        int count = 0;
        for (size_t j = 1; j < v.size(); ++j) {
            if (v.at(i) == v.at(j)) {
                ++count;
            }
        }
        if (count > count_max) {
            count_max = count;
            mode_final = v.at(i);
        }
    }
    return(mode_final);
}
```

1. **Purpose**: This function calculates the mode, which is the most frequently occurring value in the dataset.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector.
   - `int count_max = 1;`: Initializes the maximum count of occurrences.
   - `double mode_final = v.at(0);`: Assumes the first element is the mode initially.
   - Nested loops:
     - Outer loop iterates over each element.
     - Inner loop counts occurrences of the current element.
     - If a higher count is found, updates `count_max` and `mode_final`.

3. **Issues**:
   - **Inefficiency**: The nested loop results in a time complexity of O(n^2), which is inefficient for large datasets.
   - **Logic Error**: The inner loop should start from `i` instead of `1` to avoid unnecessary comparisons.

4. **Example**:
   - If `v = {1.0, 2.0, 2.0, 3.0}`, the mode is `2.0`.

### Function: `min`

```cpp
double min(vector<double> v) {
    std::sort(v.begin(), v.end());
    double min_number = v.at(0);
    return (min_number);
}
```

1. **Purpose**: This function finds the minimum value in the vector `v`.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector.
   - `double min_number = v.at(0);`: The first element of a sorted vector is the minimum.
   - `return (min_number);`: Returns the minimum value.

3. **Example**:
   - If `v = {3.0, 1.0, 2.0}`, after sorting, the minimum is `1.0`.

### Function: `max`

```cpp
double max(vector<double> v) {
    std::sort(v.begin(), v.end());
    double max_number = v.at(v.size() - 1);
    return (max_number);
}
```

1. **Purpose**: This function finds the maximum value in the vector `v`.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector.
   - `double max_number = v.at(v.size() - 1);`: The last element of a sorted vector is the maximum.
   - `return (max_number);`: Returns the maximum value.

3. **Example**:
   - If `v = {3.0, 1.0, 2.0}`, after sorting, the maximum is `3.0`.

### Function: `stdev`

```cpp
double stdev(vector<double> v) {
    double mean_n = mean(v);
    double total = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        double numerator = (v.at(i) - mean_n) * (v.at(i) - mean_n);
        total = total + numerator;
    }
    double stdev_number = sqrt(total / (v.size() - 1));
    return (stdev_number);
}
```

1. **Purpose**: This function calculates the standard deviation, which measures the amount of variation or dispersion in the dataset.

2. **Logic**:
   - `double mean_n = mean(v);`: Calls the `mean` function to get the average.
   - `double total = 0;`: Initializes `total` to accumulate squared differences.
   - Loop:
     - `(v.at(i) - mean_n) * (v.at(i) - mean_n)`: Calculates the squared difference from the mean for each element.
     - `total = total + numerator;`: Adds the squared difference to `total`.
   - `double stdev_number = sqrt(total / (v.size() - 1));`: Computes the standard deviation using the formula for sample standard deviation.
   - `return (stdev_number);`: Returns the standard deviation.

3. **Example**:
   - If `v = {1.0, 2.0, 3.0}`, the mean is `2.0`. The squared differences are `1.0, 0.0, 1.0`, and the standard deviation is `sqrt(1.0)`.

### Function: `percentile`

```cpp
double percentile(vector<double> v, double p) {
    double percentile_number;
    double percentile_final;
    double intpart;
    double fractpart;
    std::sort(v.begin(), v.end());
    percentile_number = p * (v.size() - 1);
    fractpart = modf(percentile_number, &intpart);
    if (intpart == v.size() - 1) {
        percentile_final = v.at(intpart);
        return (percentile_final);
    }
    else {
        percentile_final = v.at(intpart) + fractpart * (v.at(intpart + 1) - v.at(intpart));
        return (percentile_final);
    }
}
```

1. **Purpose**: This function calculates the p-th percentile, which is the value below which a given percentage of observations fall.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector.
   - `percentile_number = p * (v.size() - 1);`: Calculates the index for the percentile.
   - `fractpart = modf(percentile_number, &intpart);`: Splits `percentile_number` into integer and fractional parts.
   - `if (intpart == v.size() - 1)`: Checks if the index is the last element.
     - If true, returns the last element.
     - Otherwise, uses linear interpolation to calculate the percentile value.

3. **Example**:
   - If `v = {1.0, 2.0, 3.0}` and `p = 0.5`, the 50th percentile is `2.0`.

### Function: `summarize`

```cpp
vector<vector<double>> summarize(vector<double> v) {
    std::sort(v.begin(), v.end());
    vector<vector<double>> summary;
    int summary_count = 1;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v.at(i - 1) == v.at(i)) {
            summary_count++;
        }
        if (v.at(i - 1) != v.at(i)) {
            vector<double> trial;
            trial.push_back(v.at(i - 1));
            trial.push_back(summary_count);
            summary.push_back(trial);
            summary_count = 1;
        }
        if (i == v.size() - 1) {
            vector<double> trial;
            trial.push_back(v.at(i));
            trial.push_back(summary_count);
            summary.push_back(trial);
        }
    }
    return summary;
}
```

1. **Purpose**: This function summarizes the dataset by counting the frequency of each unique element.

2. **Logic**:
   - `std::sort(v.begin(), v.end());`: Sorts the vector.
   - `vector<vector<double>> summary;`: Initializes a vector of vectors to store the summary.
   - `int summary_count = 1;`: Initializes a counter for occurrences.
   - Loop:
     - Compares adjacent elements to count occurrences.
     - When a new element is encountered, stores the previous element and its count in `summary`.
     - Handles the last element separately to ensure it's included.

3. **Example**:
   - If `v = {1.0, 2.0, 2.0, 3.0}`, the summary is `{{1.0, 1}, {2.0, 2}, {3.0, 1}}`.

### Conclusion

This code provides a comprehensive toolkit for statistical analysis, but it has several areas for improvement, such as fixing logic errors, improving efficiency, and ensuring correct data types. By understanding each function's purpose and logic, you can apply these concepts to analyze datasets effectively.