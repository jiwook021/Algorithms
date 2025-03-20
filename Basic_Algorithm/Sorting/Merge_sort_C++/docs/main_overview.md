# Code Overview: main.cpp

This C++ code is a complete program that demonstrates the implementation and performance measurement of the **Merge Sort** algorithm. Let's break down its purpose, functionality, and structure in detail:

---

### **Purpose of the Code**
The primary purpose of this code is to:
1. **Generate an array of random integers**.
2. **Sort the array using the Merge Sort algorithm**.
3. **Measure and display the time taken** to perform the sorting operation.
4. **Print the array before and after sorting** to visually verify the correctness of the algorithm.

The program is designed to showcase how Merge Sort works, how it can be implemented in C++, and how to measure the performance of an algorithm using timing functions.

---

### **Main Functionality**
1. **Random Array Generation**:
   - The program generates an array of 100 random integers, each ranging from 1 to 10.
   - This is done using the `rand()` function, which is seeded with the current time to ensure different random numbers on each run.

2. **Merge Sort Algorithm**:
   - The array is sorted using the **Merge Sort** algorithm, which is a **divide-and-conquer** sorting algorithm.
   - Merge Sort works by recursively splitting the array into smaller subarrays, sorting them, and then merging them back together in sorted order.

3. **Performance Measurement**:
   - The program measures the time taken to sort the array using the `clock()` function from the `<time.h>` library.
   - This provides a way to evaluate the efficiency of the Merge Sort algorithm for the given input size.

4. **Output**:
   - The program prints the array before and after sorting to demonstrate the effect of the Merge Sort algorithm.
   - It also prints the time taken to sort the array, which is useful for performance analysis.

---

### **Algorithms Used**
1. **Merge Sort**:
   - Merge Sort is a **stable, comparison-based sorting algorithm** with a time complexity of **O(n log n)**.
   - It works by:
     - **Dividing** the array into two halves.
     - **Recursively sorting** each half.
     - **Merging** the two sorted halves into a single sorted array.

2. **Random Number Generation**:
   - The `rand()` function is used to generate random numbers.
   - The `srand(time(NULL))` function seeds the random number generator with the current time to ensure different random numbers on each run.

3. **Timing**:
   - The `clock()` function is used to measure the time taken by the program.
   - The difference between the start and end times is calculated and converted to seconds.

---

### **Overall Structure**
The code is organized into several functions, each with a specific responsibility:
1. **`PrintArray`**:
   - Prints the contents of the array to the console.

2. **`InputRandomNumber_ToArray`**:
   - Fills the array with random integers between 1 and 10.

3. **`MergeTwoArea`**:
   - Merges two sorted subarrays into a single sorted array.
   - This is a helper function for the Merge Sort algorithm.

4. **`MergeSort`**:
   - Implements the Merge Sort algorithm by recursively dividing the array and merging the sorted subarrays.

5. **`main`**:
   - The entry point of the program.
   - Coordinates the generation of the array, sorting, and performance measurement.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The program starts by seeding the random number generator and initializing the array.

2. **Random Array Generation**:
   - The `InputRandomNumber_ToArray` function fills the array with random numbers.

3. **Pre-Sort Output**:
   - The `PrintArray` function displays the unsorted array.

4. **Sorting**:
   - The `MergeSort` function is called to sort the array.
   - This function recursively divides the array and uses `MergeTwoArea` to merge the sorted subarrays.

5. **Post-Sort Output**:
   - The `PrintArray` function displays the sorted array.

6. **Performance Measurement**:
   - The program calculates the time taken to sort the array and prints it.

---

### **Problem Being Solved**
The problem being solved is **sorting an array of integers efficiently**. Merge Sort is chosen because it is a highly efficient algorithm with a guaranteed time complexity of O(n log n), making it suitable for larger datasets. The program also demonstrates how to measure the performance of an algorithm, which is a common requirement in real-world applications.

---

### **Approach Taken**
1. **Divide-and-Conquer**:
   - Merge Sort uses a divide-and-conquer approach to break the problem into smaller, more manageable subproblems.

2. **Recursion**:
   - The algorithm uses recursion to repeatedly divide the array until the base case (a single element) is reached.

3. **Merging**:
   - The sorted subarrays are merged back together in a way that maintains the sorted order.

4. **Performance Measurement**:
   - The program uses timing functions to measure the efficiency of the algorithm.

---

### **Summary**
This code is a well-structured demonstration of the Merge Sort algorithm. It generates a random array, sorts it using Merge Sort, and measures the time taken to perform the sorting operation. The program is designed to be educational, showing how Merge Sort works and how to measure its performance. It is a great example of how to implement and analyze a sorting algorithm in C++.