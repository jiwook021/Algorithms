# Code Overview: main.c

This C code is a complete program that demonstrates several fundamental programming concepts: array manipulation, random number generation, sorting algorithms, and performance measurement. Let's break down its purpose and functionality in detail:

### 1. **Main Purpose**
The code's primary purpose is to:
- Generate an array of random numbers
- Display the unsorted array
- Sort the array using the Bubble Sort algorithm
- Display the sorted array
- Measure and display the time taken to perform these operations

### 2. **Key Functionalities**
The program accomplishes its purpose through these main components:

#### a) **Array Generation**
- Creates an array of 100 integers (`ArrayLength = 100`)
- Fills the array with random numbers between 1 and 10 using `rand()`

#### b) **Sorting**
- Implements the Bubble Sort algorithm to sort the array in ascending order
- Bubble Sort repeatedly steps through the array, compares adjacent elements, and swaps them if they're in the wrong order

#### c) **Performance Measurement**
- Uses the `clock()` function to measure the time taken for the entire process
- Calculates and displays the execution time in seconds

#### d) **Output**
- Displays the array contents before and after sorting
- Shows the execution time with high precision

### 3. **Algorithms Used**
- **Random Number Generation**: Uses `rand()` with `srand(time(NULL))` for seeding to ensure different random numbers on each run
- **Bubble Sort**: A simple comparison-based sorting algorithm with O(n²) time complexity
- **Timing**: Uses `clock()` from `<time.h>` to measure CPU time

### 4. **Code Structure**
The code is well-organized into functions, each with a specific responsibility:

#### a) **Global Constant**
```c
const static uint16_t ArrayLength = 100;
```
- Defines the array size as a constant (100 elements)
- `const` ensures the value can't be changed
- `static` limits its scope to this file
- `uint16_t` specifies an unsigned 16-bit integer type

#### b) **PrintArray Function**
```c
static void PrintArray(int arr[ArrayLength])
```
- Takes an array as input
- Iterates through the array and prints each element
- Used to display array contents before and after sorting

#### c) **InputRandomNumber_ToArray Function**
```c
static void InputRandomNumber_ToArray(int arr[ArrayLength])
```
- Fills the array with random numbers between 1 and 10
- Uses `rand() % 10 + 1` to generate numbers in the desired range

#### d) **vBubbleSort Function**
```c
static void vBubbleSort(int arr[ArrayLength])
```
- Implements the Bubble Sort algorithm
- Uses nested loops to repeatedly compare and swap adjacent elements
- The outer loop controls the number of passes
- The inner loop performs the comparisons and swaps

#### e) **Main Function**
```c
int main()
```
- The program's entry point
- Performs these steps in sequence:
  1. Starts timing
  2. Seeds the random number generator
  3. Creates and fills the array
  4. Prints the unsorted array
  5. Sorts the array
  6. Prints the sorted array
  7. Stops timing and calculates duration
  8. Prints the execution time

### 5. **How It All Works Together**
1. The program starts by initializing timing and random number generation
2. It creates an array and fills it with random numbers
3. The unsorted array is displayed
4. The Bubble Sort algorithm sorts the array in place
5. The sorted array is displayed
6. Finally, the program calculates and shows how long the entire process took

### 6. **Problem Being Solved**
This code demonstrates:
- How to work with arrays in C
- How to generate and use random numbers
- Implementation of a basic sorting algorithm
- How to measure and analyze program performance

While Bubble Sort isn't the most efficient sorting algorithm, this code serves as an excellent educational tool for understanding:
- Array manipulation
- Algorithm implementation
- Performance measurement
- Program structure and organization

The code provides a complete, self-contained example that could be used as a foundation for learning more advanced programming concepts.