# Code Overview: main.c

This C code is a complete program that demonstrates the **Insertion Sort algorithm** on an array of random numbers. Let's break down its purpose, functionality, and structure in detail:

---

### **Purpose of the Code**
The purpose of this code is to:
1. Generate an array of random numbers.
2. Sort the array using the **Insertion Sort algorithm**.
3. Measure and display the time taken to perform the sorting operation.
4. Print the array before and after sorting to visualize the results.

The program is designed to:
- Showcase how Insertion Sort works.
- Provide a practical example of timing code execution in C.
- Demonstrate basic array manipulation and random number generation.

---

### **Main Functionality**
1. **Random Number Generation**:
   - The program generates an array of 100 random integers between 1 and 10 using the `rand()` function.
   - The `srand(time(NULL))` function seeds the random number generator to ensure different random numbers are generated each time the program runs.

2. **Insertion Sort**:
   - The program sorts the array using the **Insertion Sort algorithm**, which is a simple sorting algorithm that builds the final sorted array one element at a time.
   - Insertion Sort is efficient for small datasets and works well for nearly sorted arrays.

3. **Timing the Execution**:
   - The program uses the `clock()` function to measure the time taken to execute the sorting operation.
   - This is useful for understanding the performance of the algorithm.

4. **Printing the Results**:
   - The program prints the array before and after sorting to show the effect of the Insertion Sort algorithm.
   - It also prints the time taken for the sorting operation.

---

### **Algorithms Used**
1. **Insertion Sort**:
   - Insertion Sort works by dividing the array into a "sorted" and "unsorted" portion.
   - It iterates through the unsorted portion, picking one element at a time and inserting it into its correct position in the sorted portion.
   - This is done by shifting elements in the sorted portion to make space for the new element.

2. **Random Number Generation**:
   - The `rand()` function generates pseudo-random numbers.
   - The `srand(time(NULL))` function seeds the random number generator with the current time to ensure different random numbers are generated each time the program runs.

3. **Timing**:
   - The `clock()` function is used to measure the CPU time taken by the program.
   - The difference between the start and end times is divided by `CLOCKS_PER_SEC` to convert the result into seconds.

---

### **Overall Structure**
The code is structured into several functions, each with a specific responsibility:
1. **`Insertionsort()`**:
   - Implements the Insertion Sort algorithm.
   - Takes an array and its length as input and sorts the array in place.

2. **`PrintArray()`**:
   - Prints the contents of the array to the console.

3. **`InputRandomNumber_ToArray()`**:
   - Fills the array with random numbers between 1 and 10.

4. **`main()`**:
   - The entry point of the program.
   - Seeds the random number generator, generates the array, prints it, sorts it, prints the sorted array, and measures the time taken.

---

### **How the Parts Work Together**
1. The `main()` function initializes the program by:
   - Seeding the random number generator.
   - Creating an array of 100 elements.
   - Filling the array with random numbers using `InputRandomNumber_ToArray()`.

2. The program prints the unsorted array using `PrintArray()`.

3. The `Insertionsort()` function is called to sort the array.

4. The sorted array is printed again using `PrintArray()`.

5. The program calculates and prints the time taken for the sorting operation using `clock()`.

---

### **Problem Being Solved**
The problem being solved is:
- Sorting an array of random numbers efficiently using the Insertion Sort algorithm.
- Demonstrating how to measure the performance of a sorting algorithm in terms of execution time.

---

### **Approach Taken**
1. **Random Number Generation**:
   - The program uses `rand()` to generate random numbers and `srand(time(NULL))` to ensure randomness.

2. **Sorting**:
   - The Insertion Sort algorithm is implemented in the `Insertionsort()` function.
   - It sorts the array in place, meaning it modifies the original array without requiring additional memory.

3. **Timing**:
   - The `clock()` function is used to measure the time taken for the sorting operation.
   - This provides a practical way to evaluate the performance of the algorithm.

4. **Output**:
   - The program prints the array before and after sorting to show the results.
   - It also prints the time taken for the sorting operation.

---

### **Summary**
This code is a well-structured demonstration of:
- Random number generation.
- The Insertion Sort algorithm.
- Timing code execution.
- Basic array manipulation and printing.

It serves as a practical example for understanding sorting algorithms, random number generation, and performance measurement in C.