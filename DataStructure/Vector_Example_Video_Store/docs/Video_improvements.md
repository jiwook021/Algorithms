# Suggested Improvements: Video.cpp

This code is functional and demonstrates good object-oriented principles, but there are several areas where it can be improved for **performance**, **readability**, **maintainability**, **error handling**, and adherence to **best practices**. Below are detailed suggestions, along with explanations and code examples for each improvement.

---

### **1. Use `const` for Getter Methods**
#### **Why:**
- Getters should not modify the object’s state. Marking them as `const` ensures they can be called on `const` objects and prevents accidental modifications.
- This improves **safety** and **readability**.

#### **How:**
Add the `const` keyword to the getter methods:
```cpp
string Videotape::getNAME_OF_TAPE() const { return NAME_OF_TAPE; }
string Videotape::getTYPE_OF_MOVIE() const { return TYPE_OF_MOVIE; }
string Videotape::getRELEASED_DATE() const { return RELEASED_DATE; }
string Videotape::getCOST() const { return COST; }
string Videotape::getAVAILABILITY() const { return AVAILABILITY; }
string Videotape::getLOCATION() const { return LOCATION; }
```

---

### **2. Use Member Initialization Lists in the Constructor**
#### **Why:**
- Member initialization lists are more efficient than assigning values in the constructor body. They directly initialize member variables instead of first default-constructing them and then assigning values.
- This improves **performance** and is considered a **best practice**.

#### **How:**
Replace the constructor body with an initialization list:
```cpp
Videotape::Videotape(string name_of_tape, string type_of_movie, string released_date, string cost, string availability, string location)
    : NAME_OF_TAPE(name_of_tape),
      TYPE_OF_MOVIE(type_of_movie),
      RELEASED_DATE(released_date),
      COST(cost),
      AVAILABILITY(availability),
      LOCATION(location) {}
```

---

### **3. Use `enum` for `AVAILABILITY`**
#### **Why:**
- Using a string for `AVAILABILITY` (e.g., `"Available"`, `"Rented"`) is error-prone because it relies on exact string matching. An `enum` ensures only valid values are used.
- This improves **type safety** and **maintainability**.

#### **How:**
Define an `enum` for availability:
```cpp
enum class Availability { Available, Rented };

class Videotape {
private:
    Availability AVAILABILITY;
public:
    void setAVAILABILITY(Availability e) { AVAILABILITY = e; }
    Availability getAVAILABILITY() const { return AVAILABILITY; }
};
```

Update the `Print` method to handle the `enum`:
```cpp
void Videotape::Print() {
    cout << "\n\nName: " << NAME_OF_TAPE << endl;
    cout << "Type of movie: " << TYPE_OF_MOVIE << endl;
    cout << "Released date : " << RELEASED_DATE << endl;
    cout << "Cost: " << COST << endl;
    cout << "AVAILABILITY: " << (AVAILABILITY == Availability::Available ? "Available" : "Rented") << endl;
    cout << "Location: " << LOCATION << "\n\n";
}
```

---

### **4. Use `double` for `COST`**
#### **Why:**
- Storing the cost as a `string` (e.g., `"$5.99"`) makes it difficult to perform calculations (e.g., total rental cost). Using a `double` is more appropriate for numeric values.
- This improves **performance** and **usability**.

#### **How:**
Change the `COST` member variable and related methods:
```cpp
private:
    double COST;

public:
    Videotape(string name_of_tape, string type_of_movie, string released_date, double cost, Availability availability, string location)
        : NAME_OF_TAPE(name_of_tape), TYPE_OF_MOVIE(type_of_movie), RELEASED_DATE(released_date), COST(cost), AVAILABILITY(availability), LOCATION(location) {}

    void setCOST(double d) { COST = d; }
    double getCOST() const { return COST; }
```

Update the `Print` method to format the cost:
```cpp
void Videotape::Print() {
    cout << "\n\nName: " << NAME_OF_TAPE << endl;
    cout << "Type of movie: " << TYPE_OF_MOVIE << endl;
    cout << "Released date : " << RELEASED_DATE << endl;
    cout << "Cost: $" << fixed << setprecision(2) << COST << endl; // Format to 2 decimal places
    cout << "AVAILABILITY: " << (AVAILABILITY == Availability::Available ? "Available" : "Rented") << endl;
    cout << "Location: " << LOCATION << "\n\n";
}
```

---

### **5. Add Input Validation**
#### **Why:**
- The current code does not validate input, which could lead to invalid data (e.g., negative cost, empty strings). Input validation ensures data integrity.
- This improves **robustness** and **error handling**.

#### **How:**
Add validation to setters:
```cpp
void Videotape::setCOST(double d) {
    if (d < 0) {
        throw invalid_argument("Cost cannot be negative.");
    }
    COST = d;
}

void Videotape::setNAME_OF_TAPE(const string& a) {
    if (a.empty()) {
        throw invalid_argument("Name cannot be empty.");
    }
    NAME_OF_TAPE = a;
}
```

---

### **6. Use `const` References for String Parameters**
#### **Why:**
- Passing strings by value creates unnecessary copies, which is inefficient. Passing by `const` reference avoids copying and ensures the original string is not modified.
- This improves **performance** and **safety**.

#### **How:**
Update the constructor and setters:
```cpp
Videotape::Videotape(const string& name_of_tape, const string& type_of_movie, const string& released_date, double cost, Availability availability, const string& location)
    : NAME_OF_TAPE(name_of_tape), TYPE_OF_MOVIE(type_of_movie), RELEASED_DATE(released_date), COST(cost), AVAILABILITY(availability), LOCATION(location) {}

void Videotape::setNAME_OF_TAPE(const string& a) { NAME_OF_TAPE = a; }
```

---

### **7. Improve the `Print` Method**
#### **Why:**
- The `Print` method hardcodes the output format, making it inflexible. Using a more dynamic approach (e.g., returning a formatted string) allows for reuse in different contexts (e.g., GUI, file output).
- This improves **flexibility** and **maintainability**.

#### **How:**
Return a formatted string instead of printing directly:
```cpp
string Videotape::ToString() const {
    ostringstream oss;
    oss << "\n\nName: " << NAME_OF_TAPE << endl
        << "Type of movie: " << TYPE_OF_MOVIE << endl
        << "Released date : " << RELEASED_DATE << endl
        << "Cost: $" << fixed << setprecision(2) << COST << endl
        << "AVAILABILITY: " << (AVAILABILITY == Availability::Available ? "Available" : "Rented") << endl
        << "Location: " << LOCATION << "\n\n";
    return oss.str();
}
```

Usage:
```cpp
cout << myTape.ToString();
```

---

### **8. Use `std::optional` for Optional Fields**
#### **Why:**
- Some fields (e.g., `RELEASED_DATE`, `LOCATION`) might not always have a value. Using `std::optional` makes it clear when a field is optional and avoids using placeholder values (e.g., empty strings).
- This improves **clarity** and **type safety**.

#### **How:**
Include `<optional>` and update the class:
```cpp
#include <optional>

class Videotape {
private:
    std::optional<string> RELEASED_DATE;
    std::optional<string> LOCATION;
public:
    void setRELEASED_DATE(const std::optional<string>& c) { RELEASED_DATE = c; }
    std::optional<string> getRELEASED_DATE() const { return RELEASED_DATE; }
};
```

Update the `Print` method to handle optional fields:
```cpp
void Videotape::Print() const {
    cout << "\n\nName: " << NAME_OF_TAPE << endl;
    cout << "Type of movie: " << TYPE_OF_MOVIE << endl;
    cout << "Released date : " << (RELEASED_DATE ? *RELEASED_DATE : "N/A") << endl;
    cout << "Cost: $" << fixed << setprecision(2) << COST << endl;
    cout << "AVAILABILITY: " << (AVAILABILITY == Availability::Available ? "Available" : "Rented") << endl;
    cout << "Location: " << (LOCATION ? *LOCATION : "N/A") << "\n\n";
}
```

---

### **9. Add a Destructor (If Needed)**
#### **Why:**
- If the class manages dynamic memory (e.g., pointers), a destructor is necessary to avoid memory leaks. In this case, it’s not needed, but it’s good to be aware of this principle.

---

### **10. Follow Naming Conventions**
#### **Why:**
- Using consistent naming conventions (e.g., `snake_case` or `camelCase`) improves **readability** and **maintainability**.

#### **How:**
Rename member variables to use `snake_case`:
```cpp
private:
    string name_of_tape;
    string type_of_movie;
    string released_date;
    double cost;
    Availability availability;
    string location;
```

---

### **Final Improved Code**
Here’s how the improved class might look:
```cpp
#include <iostream>
#include <string>
#include <optional>
#include <sstream>
#include <iomanip>

enum class Availability { Available, Rented };

class Videotape {
private:
    std::string name_of_tape;
    std::string type_of_movie;
    std::optional<std::string> released_date;
    double cost;
    Availability availability;
    std::optional<std::string> location;

public:
    Videotape(const std::string& name_of_tape, const std::string& type_of_movie, const std::optional<std::string>& released_date, double cost, Availability availability, const std::optional<std::string>& location)
        : name_of_tape(name_of_tape), type_of_movie(type_of_movie), released_date(released_date), cost(cost), availability(availability), location(location) {}

    void set_name_of_tape(const std::string& a) { name_of_tape = a; }
    void set_type_of_movie(const std::string& b) { type_of_movie = b; }
    void set_released_date(const std::optional<std::string>& c) { released_date = c; }
    void set_cost(double d) { cost = d; }
    void set_availability(Availability e) { availability = e; }
    void set_location(const std::optional<std::string>& f) { location = f; }

    std::string get_name_of_tape() const { return name_of_tape; }
    std::string get_type_of_movie() const { return type_of_movie; }
    std::optional<std::string> get_released_date() const { return released_date; }
    double get_cost() const { return cost; }
    Availability get_availability() const { return availability; }
    std::optional<std::string> get_location() const { return location; }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "\n\nName: " << name_of_tape << std::endl
            << "Type of movie: " << type_of_movie << std::endl
            << "Released date : " << (released_date ? *released_date : "N/A") << std::endl
            << "Cost: $" << std::fixed << std::setprecision(2) << cost << std::endl
            << "AVAILABILITY: " << (availability == Availability::Available ? "Available" : "Rented") << std::endl
            << "Location: " << (location ? *location : "N/A") << "\n\n";
        return oss.str();
    }
};
```

---

These improvements make the code more **efficient**, **readable**, **maintainable**, and **robust**, while adhering to modern C++ best practices.