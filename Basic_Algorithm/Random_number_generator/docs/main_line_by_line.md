# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Header Comments and Includes**
```cpp
/**
 * @file main.cpp
 * @author Jiwook Kim (Jiwook021@gmail.com)
 * @brief Find the number of Random generated Value
 * @version 0.1
 * @date 2022-08-27
 * @copyright Copyright (c) 2022
 */

//Include Files
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cstdint>
```

#### **What It Does**
- The header comments describe the file’s purpose, author, version, and date.
- The `#include` directives bring in necessary libraries:
  - `<iostream>`: For input/output (e.g., printing to the console).
  - `<vector>`: For using dynamic arrays (`std::vector`).
  - `<array>`: For using fixed-size arrays (`std::array`).
  - `<random>`: For generating random numbers.
  - `<cstdint>`: For fixed-width integer types like `uint8_t` and `uint16_t`.

#### **Why It’s Used**
- The comments provide context for the code, making it easier for others (or yourself) to understand later.
- The libraries are included to provide the tools needed for the program (e.g., random number generation, dynamic arrays).

---

### **2. Global Variable**
```cpp
const uint8_t RESULT_ARRAY_SIZE = 10;
```

#### **What It Does**
- Defines a constant (`RESULT_ARRAY_SIZE`) with a value of 10.
- `uint8_t` is an unsigned 8-bit integer (range: 0 to 255).

#### **Why It’s Used**
- This constant is used to define the size of the array that will store the counts of each number (0 through 9). Using a constant makes the code easier to maintain (e.g., if the range changes, you only need to update this one value).

---

### **3. Function Prototypes**
```cpp
void RandomGenerator(const uint8_t, const uint8_t, std::vector<int>&, const uint16_t);
void PrintVector(std::vector<int>);
void CountOccurrenceOfNum(const std::vector<int>, std::array<uint16_t,10> &);
void PrintResultArray(const std::array<uint16_t,10>);
```

#### **What It Does**
- Declares the functions that will be defined later. This tells the compiler what functions to expect and their signatures (return type, name, and parameters).

#### **Why It’s Used**
- Function prototypes allow the compiler to check for errors (e.g., incorrect function calls) before the functions are defined. They also make the code more organized.

---

### **4. RandomGenerator Function**
```cpp
void RandomGenerator(const uint8_t Lower_bound, const uint8_t Upper_bound, std::vector<int> &random_numbers, const uint16_t VectorLength) 
{
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(Lower_bound, Upper_bound);
    
    for (uint16_t i = 0; i < VectorLength; i++)
    {
        random_numbers.push_back(dis(gen));
    }
}
```

#### **What It Does**
- Generates random numbers between `Lower_bound` and `Upper_bound` and stores them in a vector.

#### **Step-by-Step Breakdown**
1. **Random Number Setup**:
   - `std::random_device rd;`: Creates a random number generator seed. This ensures the numbers are truly random.
   - `std::mt19937 gen(rd());`: Initializes the Mersenne Twister random number generator with the seed.
   - `std::uniform_int_distribution<int> dis(Lower_bound, Upper_bound);`: Defines a uniform distribution for integers between `Lower_bound` and `Upper_bound`.

2. **Generating Numbers**:
   - A `for` loop runs `VectorLength` times.
   - In each iteration, `dis(gen)` generates a random number, and `push_back()` adds it to the vector.

#### **Why It’s Used**
- The Mersenne Twister (`std::mt19937`) is used because it produces high-quality random numbers.
- The uniform distribution ensures each number in the range has an equal chance of being selected.

---

### **5. PrintVector Function**
```cpp
void PrintVector(std::vector<int> vectors) 
{
    std::cout<< "Stored Value " << std::endl;     
    std::vector<int>::iterator iter = vectors.begin();
    for( ; iter != vectors.end(); iter++)
    {
        std::cout << *iter << " ";
    }
}
```

#### **What It Does**
- Prints all the numbers stored in the vector.

#### **Step-by-Step Breakdown**
1. **Iterator Setup**:
   - `std::vector<int>::iterator iter = vectors.begin();`: Creates an iterator pointing to the first element of the vector.
2. **Loop Through Vector**:
   - The `for` loop iterates through the vector using the iterator.
   - `*iter` dereferences the iterator to access the current element.
   - `std::cout` prints the element followed by a space.

#### **Why It’s Used**
- Iterators provide a way to traverse containers like vectors. They are flexible and work with all standard containers.

---

### **6. CountOccurrenceOfNum Function**
```cpp
void CountOccurrenceOfNum(const std::vector<int> Vectors, std::array<uint16_t,RESULT_ARRAY_SIZE> &Result)
{
    for (uint16_t i=0; i < Vectors.size();i++) 
    {  
        Result[Vectors[i]]++;  
    }
}
```

#### **What It Does**
- Counts how many times each number (0 through 9) appears in the vector.

#### **Step-by-Step Breakdown**
1. **Loop Through Vector**:
   - The `for` loop iterates through each element in the vector.
2. **Count Occurrences**:
   - `Result[Vectors[i]]++`: Uses the number itself as an index into the `Result` array and increments the count.

#### **Why It’s Used**
- This approach is efficient because it directly maps each number to its count using array indexing.

---

### **7. PrintResultArray Function**
```cpp
void PrintResultArray(const std::array<uint16_t, RESULT_ARRAY_SIZE> Result)
{
    std::cout << "\n\nOccurrence" << std::endl;
    uint16_t idx = 0; 
    for (auto i: Result)
    {
        std::cout << "Result[" << idx << "] :" << i << std::endl;
        idx++;
    }
}
```

#### **What It Does**
- Prints the counts of each number.

#### **Step-by-Step Breakdown**
1. **Loop Through Array**:
   - A range-based `for` loop iterates through the `Result` array.
2. **Print Counts**:
   - `std::cout` prints the index (number) and its corresponding count.

#### **Why It’s Used**
- Range-based loops are concise and easy to read for iterating over containers.

---

### **8. Main Function**
```cpp
int main() 
{
    const uint16_t VectorLength = 100; 
    std::vector<int> Vector_random_numbers;  
    Vector_random_numbers.reserve((const uint16_t) VectorLength);  

    RandomGenerator((const uint8_t) 0, (const uint8_t) 9, (std::vector<int>&)Vector_random_numbers, (const uint16_t) VectorLength);  
    PrintVector((std::vector<int>&) Vector_random_numbers);  

    std::array<uint16_t, RESULT_ARRAY_SIZE> Result = { 0 };      
    CountOccurrenceOfNum((const std::vector<int>&) Vector_random_numbers, (std::array<uint16_t,RESULT_ARRAY_SIZE> &) Result);  
    PrintResultArray((const std::array<uint16_t,RESULT_ARRAY_SIZE> &) Result);  

    Vector_random_numbers.clear();
    Vector_random_numbers.shrink_to_fit();

    return 0; 
}
```

#### **What It Does**
- Coordinates the execution of the program.

#### **Step-by-Step Breakdown**
1. **Initialize Vector**:
   - `Vector_random_numbers.reserve(VectorLength);`: Preallocates memory for the vector to improve performance.
2. **Generate Random Numbers**:
   - Calls `RandomGenerator` to fill the vector with 100 random numbers.
3. **Print Numbers**:
   - Calls `PrintVector` to display the generated numbers.
4. **Count Occurrences**:
   - Calls `CountOccurrenceOfNum` to count how many times each number appears.
5. **Print Results**:
   - Calls `PrintResultArray` to display the counts.
6. **Cleanup**:
   - `clear()` removes all elements from the vector.
   - `shrink_to_fit()` frees unused memory.

#### **Why It’s Used**
- The `main` function ties everything together and ensures the program runs in the correct order.

---

### **Diagram: Program Flow**
```
main()
  |
  |--> RandomGenerator() --> Fills vector with random numbers
  |
  |--> PrintVector() --> Displays the numbers
  |
  |--> CountOccurrenceOfNum() --> Counts occurrences of each number
  |
  |--> PrintResultArray() --> Displays the counts
  |
  |--> Cleanup (clear() and shrink_to_fit())
```

---

This concludes the detailed explanation. Let me know if you’d like to proceed with the next question!