# Suggested Improvements: stats.cpp

The `stats.cpp` code provides a basic framework for statistical analysis, but there are several areas where improvements can be made to enhance performance, readability, maintainability, and correctness. Let's explore these improvements in detail:

### 1. **Use of `const` References**

**Why**: Passing vectors by value (as done in the current code) creates a copy of the vector each time a function is called, which is inefficient for large datasets. Using `const` references avoids unnecessary copying and protects the data from being modified within the function.

**How**: Change function parameters from `vector<double> v` to `const vector<double>& v`.

**Example**:
```cpp
int count(const vector<double>& v) {
    return v.size();
}
```

### 2. **Correct Data Types**

**Why**: The `mean` function uses `int` for `sum_number` and `mean_number`, which can lead to incorrect results due to integer division. Using `double` ensures that decimal values are handled correctly.

**How**: Change `int` to `double` in the `mean` function.

**Example**:
```cpp
double mean(const vector<double>& v) {
    double sum_number = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum_number += v.at(i);
    }
    return sum_number / v.size();
}
```

### 3. **Fix Median Calculation**

**Why**: The median calculation is incorrect due to the use of `+ 0.5` and `- 0.5`. The correct approach is to average the two middle elements for even-sized vectors.

**How**: Correct the logic to calculate the median properly.

**Example**:
```cpp
double median(vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t mid = v.size() / 2;
    if (v.size() % 2 == 0) {
        return (v[mid - 1] + v[mid]) / 2.0;
    } else {
        return v[mid];
    }
}
```

### 4. **Optimize Mode Calculation**

**Why**: The current mode calculation is inefficient with a time complexity of O(n^2). Using a hash map (unordered_map) can reduce this to O(n).

**How**: Use `unordered_map` to count occurrences of each element.

**Example**:
```cpp
#include <unordered_map>

double mode(const vector<double>& v) {
    std::unordered_map<double, int> frequency;
    for (double num : v) {
        frequency[num]++;
    }
    int max_count = 0;
    double mode_value = v[0];
    for (const auto& pair : frequency) {
        if (pair.second > max_count) {
            max_count = pair.second;
            mode_value = pair.first;
        }
    }
    return mode_value;
}
```

### 5. **Avoid Unnecessary Sorting**

**Why**: Sorting is an O(n log n) operation. Functions like `min` and `max` don't need sorting to find the smallest or largest element.

**How**: Use `std::min_element` and `std::max_element` instead.

**Example**:
```cpp
#include <algorithm>

double min(const vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
}

double max(const vector<double>& v) {
    return *std::max_element(v.begin(), v.end());
}
```

### 6. **Error Handling**

**Why**: The code does not handle edge cases, such as empty vectors, which can lead to runtime errors.

**How**: Add checks to handle empty vectors gracefully.

**Example**:
```cpp
double mean(const vector<double>& v) {
    if (v.empty()) {
        throw std::invalid_argument("Vector is empty");
    }
    double sum_number = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum_number += v.at(i);
    }
    return sum_number / v.size();
}
```

### 7. **Improve Readability and Consistency**

**Why**: Consistent naming and formatting improve readability and maintainability.

**How**: Use consistent naming conventions and simplify expressions.

**Example**:
```cpp
double sum(const vector<double>& v) {
    double total = 0;
    for (double num : v) {
        total += num;
    }
    return total;
}
```

### 8. **Use Standard Library Functions**

**Why**: The C++ Standard Library provides efficient and well-tested functions for common tasks.

**How**: Use `std::accumulate` for summing elements.

**Example**:
```cpp
#include <numeric>

double sum(const vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}
```

### 9. **Refactor `summarize` Function**

**Why**: The current implementation is complex and can be simplified using a map.

**How**: Use `unordered_map` to count occurrences and then convert to a vector.

**Example**:
```cpp
vector<vector<double>> summarize(const vector<double>& v) {
    std::unordered_map<double, int> frequency;
    for (double num : v) {
        frequency[num]++;
    }
    vector<vector<double>> summary;
    for (const auto& pair : frequency) {
        summary.push_back({pair.first, static_cast<double>(pair.second)});
    }
    return summary;
}
```

By implementing these improvements, the code becomes more efficient, readable, and robust, making it easier to maintain and extend in the future.