# Step-by-Step Explanation: p1_library.cpp

Let's dive into the `p1_library.cpp` code step-by-step, explaining each part thoroughly. We'll break down the code into manageable sections and explain everything in simple terms, ensuring that even someone new to programming can follow along.

### File Header Comments

```cpp
/* p1_library.cpp
 *
 * Libraries needed for EECS 280 project 1
 * Project UID 5366c7e2b77742d5b2142097e51561a5
 *
 * by Andrew DeOrio <awdeorio@umich.edu>
 * 2015-04-28
 */
```

1. **Purpose**: This section provides metadata about the file. It includes the filename, a brief description, a project identifier, the author's name, and the date.
2. **Why It's Here**: Comments like these are crucial for documentation. They help anyone reading the code understand its purpose and origin. This is especially useful in collaborative environments or when revisiting code after some time.

### Header Guards

```cpp
#ifndef CSVSTREAM_H
#define CSVSTREAM_H
```

1. **What It Does**: These lines are called header guards. They prevent the contents of the file from being included more than once in a single compilation.
2. **Logic**: 
   - `#ifndef CSVSTREAM_H`: Checks if `CSVSTREAM_H` has not been defined yet.
   - `#define CSVSTREAM_H`: If not, it defines `CSVSTREAM_H`.
   - The code between `#ifndef` and `#endif` is included only if `CSVSTREAM_H` is not already defined.
3. **Why It's Used**: In C++, including the same header file multiple times can lead to errors due to redefinition. Header guards ensure that the file's contents are only included once, preventing such errors.

### Include Statements

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <exception>
```

1. **What It Does**: These lines include standard library headers that provide various functionalities.
2. **Breakdown**:
   - `<iostream>`: For input and output operations.
   - `<fstream>`: For file stream operations, allowing reading from and writing to files.
   - `<sstream>`: For string stream operations, useful for parsing strings.
   - `<cassert>`: For using assertions, which are checks that can help catch errors during development.
   - `<string>`: For handling strings.
   - `<vector>`: For using dynamic arrays (vectors).
   - `<map>`: For using associative arrays (maps), which store key-value pairs.
   - `<regex>`: For regular expressions, useful for pattern matching within strings.
   - `<exception>`: For handling exceptions, which are errors that can be caught and managed.
3. **Why It's Used**: Including these libraries provides the necessary tools to perform various operations required by the CSV parsing functionality.

### Custom Exception Class

```cpp
class csvstream_exception : public std::exception {
public:
    const char* what() const noexcept override {
        return msg.c_str();
    }
    const std::string msg;
    csvstream_exception(const std::string& msg) : msg(msg) {};
};
```

1. **What It Does**: Defines a custom exception class named `csvstream_exception`.
2. **Breakdown**:
   - **Inheritance**: `: public std::exception` means this class inherits from `std::exception`, the base class for all standard exceptions in C++.
   - **Constructor**: `csvstream_exception(const std::string& msg) : msg(msg) {}` initializes the `msg` member with the provided message.
   - **what() Method**: This method returns the error message. `const char* what() const noexcept override` means it returns a C-style string, doesn't modify the object, doesn't throw exceptions, and overrides the base class method.
3. **Why It's Used**: Custom exceptions allow for more specific error handling. By creating a `csvstream_exception`, the library can provide detailed error messages specific to CSV parsing issues.

### CSV Stream Class Declaration

```cpp
class csvstream {
public:
    csvstream(const std::string& filename, char delimiter = ',', bool strict = true);
    csvstream(std::istream& is, char delimiter = ',', bool strict = true);
    ~csvstream();
    explicit operator bool() const;
    std::vector<std::string> getheader() const;
    csvstream& operator>> (std::map<std::string, std::string>& row);
    csvstream& operator>> (std::vector<std::pair<std::string, std::string> >& row);

private:
    std::string filename;
    std::ifstream fin;
    std::istream& is;
};
```

1. **What It Does**: Declares the `csvstream` class, which provides the interface for parsing CSV files.
2. **Public Members**:
   - **Constructors**: Initialize the class with either a filename or an input stream. They also allow specifying a delimiter and whether to enforce strict parsing.
   - **Destructor**: Cleans up resources when an object of this class is destroyed.
   - **Conversion Operator**: `explicit operator bool() const` allows checking the stream's state (e.g., if it's still good for reading).
   - **getheader()**: Returns the header of the CSV file as a vector of strings.
   - **operator>> Overloads**: These functions read a row from the CSV file into either a map or a vector of pairs.
3. **Private Members**:
   - **filename**: Stores the name of the file being processed.
   - **fin**: An input file stream used when reading from a file.
   - **is**: A reference to the input stream being used, whether it's from a file or another source.
4. **Why It's Used**: The class encapsulates all the functionality needed to parse CSV files, providing a clean and easy-to-use interface for users.

### Explanation of Concepts

- **Class**: A blueprint for creating objects. It encapsulates data and functions that operate on that data.
- **Constructor**: A special function that initializes objects of a class.
- **Destructor**: A special function that cleans up when an object is destroyed.
- **Operator Overloading**: Allows defining custom behavior for operators (like `>>`) when used with class objects.
- **Inheritance**: A way to form new classes using classes that have already been defined.

### Why These Techniques Are Used

- **Encapsulation**: By using a class, the library hides the complex details of CSV parsing and provides a simple interface.
- **Exception Handling**: Custom exceptions allow the library to communicate specific errors to the user, making it easier to diagnose issues.
- **Operator Overloading**: Makes the library intuitive to use, as it mimics the behavior of standard input streams in C++.

### Summary

The `p1_library.cpp` file defines a library for parsing CSV files in C++. It uses classes to encapsulate functionality, exceptions for error handling, and operator overloading for intuitive usage. By understanding each part of the code, we can see how it provides a robust solution for reading and processing CSV data in C++ applications.