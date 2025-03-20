# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <any>
#include <algorithm>
#include <stdexcept>
#include <sstream>  // Instead of format
#include <atomic>
#include <thread>
```

#### **What It Does**
These are **header files** that provide functionality for the program. Each header file includes pre-written code for specific tasks:
- `<iostream>`: For input/output (e.g., printing to the console).
- `<string>`: For working with text (e.g., storing fact names like "temperature").
- `<unordered_map>`: For storing key-value pairs (e.g., mapping fact names to their values).
- `<mutex>`: For thread safety (e.g., preventing multiple threads from modifying data at the same time).

#### **Why It’s Used**
- These headers provide reusable tools so we don’t have to write everything from scratch.
- For example, `<unordered_map>` is used to store facts efficiently, and `<mutex>` ensures the program works correctly in multi-threaded environments.

---

### **2. Class Fact**
```cpp
class Fact {
public:
    using FactValue = std::variant<bool, int, double, std::string>;
    
    Fact(std::string name, FactValue value)
        : name_(std::move(name)), value_(std::move(value)) {}
    
    [[nodiscard]] const std::string& getName() const {
        return name_;
    }
    
    [[nodiscard]] const FactValue& getValue() const {
        return value_;
    }
    
    void setValue(FactValue value) {
        value_ = std::move(value);
    }
    
    [[nodiscard]] std::string toString() const {
        std::string result = name_ + " = ";
        
        std::visit([&result](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::string>) {
                result += "\"" + val + "\"";
            } else if constexpr (std::is_same_v<T, bool>) {
                result += val ? "true" : "false";
            } else {
                result += std::to_string(val);
            }
        }, value_);
        
        return result;
    }

private:
    std::string name_;
    FactValue value_;
};
```

#### **What It Does**
This class represents a **fact**, which is a piece of information with a name and value. For example:
- Name: `"temperature"`, Value: `102`
- Name: `"diagnosis"`, Value: `"flu"`

#### **Key Components**
1. **`FactValue`**:
   - A `std::variant` that can store one of several types: `bool`, `int`, `double`, or `std::string`.
   - Think of it as a box that can hold different types of data.

2. **Constructor**:
   - Initializes a fact with a name and value.
   - Example: `Fact("temperature", 102)` creates a fact with name `"temperature"` and value `102`.

3. **`getName()` and `getValue()`**:
   - These are **getter methods** that return the name and value of the fact.
   - Example: If `fact.getName()` is called, it returns `"temperature"`.

4. **`setValue()`**:
   - Updates the value of the fact.
   - Example: `fact.setValue(103)` changes the value to `103`.

5. **`toString()`**:
   - Converts the fact to a string for display.
   - Example: `fact.toString()` returns `"temperature = 102"`.

#### **Why It’s Used**
- The `Fact` class encapsulates all the information about a fact, making it easy to store, retrieve, and manipulate.
- Using `std::variant` allows the fact to hold different types of data, making the system flexible.

---

### **3. Class FactBase**
```cpp
class FactBase {
public:
    bool addFact(const Fact& fact) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& name = fact.getName();
        auto [it, inserted] = facts_.try_emplace(name, fact);
        
        if (!inserted) {
            it->second = fact;
        }
        
        return inserted;
    }
    
    bool removeFact(const std::string& factName) {
        std::lock_guard<std::mutex> lock(mutex_);
        return facts_.erase(factName) > 0;
    }
    
    [[nodiscard]] bool hasFact(const std::string& factName) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return facts_.contains(factName);
    }
    
    [[nodiscard]] std::optional<Fact> getFact(const std::string& factName) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = facts_.find(factName);
        if (it != facts_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Fact> facts_;
};
```

#### **What It Does**
This class represents a **collection of facts** (working memory). It provides methods to add, remove, and query facts.

#### **Key Components**
1. **`addFact()`**:
   - Adds a new fact or updates an existing one.
   - Example: `factBase.addFact(Fact("temperature", 102))` adds the fact to the collection.

2. **`removeFact()`**:
   - Removes a fact by its name.
   - Example: `factBase.removeFact("temperature")` removes the fact with name `"temperature"`.

3. **`hasFact()`**:
   - Checks if a fact exists.
   - Example: `factBase.hasFact("temperature")` returns `true` if the fact exists.

4. **`getFact()`**:
   - Retrieves a fact by its name.
   - Example: `factBase.getFact("temperature")` returns the fact if it exists.

5. **Thread Safety**:
   - A `std::mutex` is used to ensure that only one thread can modify the fact base at a time.
   - This prevents **data races**, where multiple threads try to modify the same data simultaneously.

#### **Why It’s Used**
- The `FactBase` class acts as the **working memory** of the system, storing all the facts.
- Thread safety ensures the system works correctly in multi-threaded environments.

---

### **4. Main Function**
```cpp
int main() {
    try {
        // Run tests
        test::runSimpleTest();
        test::runMedicalDiagnosisTest();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

#### **What It Does**
This is the **entry point** of the program. It runs tests to demonstrate the system’s functionality.

#### **Key Components**
1. **`try-catch` Block**:
   - The `try` block runs the tests.
   - If an error occurs, the `catch` block handles it and prints an error message.

2. **Tests**:
   - `test::runSimpleTest()` and `test::runMedicalDiagnosisTest()` are hypothetical functions that test the system.
   - Example: A test might add facts like `"temperature = 102"` and check if the system correctly derives `"fever = true"`.

#### **Why It’s Used**
- The `main` function is where the program starts execution.
- Testing ensures the system works as expected.

---

### **Summary**
This code implements a rule-based system that:
1. Stores facts (e.g., `"temperature = 102"`).
2. Applies rules (e.g., `"If temperature > 100, then fever = true"`).
3. Derives new facts (e.g., `"fever = true"`).

The system is modular, thread-safe, and flexible, making it suitable for applications like medical diagnosis or business rule engines.