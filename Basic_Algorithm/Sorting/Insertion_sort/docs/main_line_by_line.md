# Step-by-Step Explanation: main.c

Let’s break down the code **line by line** and **function by function**, explaining everything in detail. I’ll use simple language, examples, and diagrams to make it easy to understand, even for beginners.

---

### **1. Header Files**
```c
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <stdint.h>
```
#### What it does:
- These are **header files** that provide functions and definitions used in the program.
- Think of them as toolboxes that give the program access to specific tools (functions and data types).

#### Breakdown:
1. **`<stdio.h>`**: Provides input/output functions like `printf()` for printing to the console.
2. **`<stdlib.h>`**: Provides functions for memory allocation, random number generation (`rand()` and `srand()`), and other utilities.
3. **`<time.h>`**: Provides functions for working with time, such as `clock()` for timing and `time()` for seeding the random number generator.
4. **`<stdint.h>`**: Provides fixed-width integer types like `uint8_t`, which is an unsigned 8-bit integer (values from 0 to 255).

#### Why it’s used:
- These headers are necessary because the program uses functions like `printf()`, `rand()`, `srand()`, and `clock()`, as well as the `uint8_t` data type.

---

### **2. Global Constant**
```c
static const uint8_t ArrayLength = 100;
```
#### What it does:
- Defines a **constant** named `ArrayLength` with a value of 100.
- The `static` keyword means this constant is only visible within this file (not accessible from other files).
- The `const` keyword means the value cannot be changed.

#### Why it’s used:
- This constant is used to define the size of the array. Using a constant makes the code easier to maintain because if you want to change the array size, you only need to update this one line.

---

### **3. Insertionsort Function**
```c
static void Insertionsort(uint8_t arr[], int n)
{
    static uint8_t insData;     
    int j; 
    int i;
    for(i = 1; i < n; i++)
    {
        insData = arr[i];
        for(j = i - 1; j >= 0; j--)
        {
            if(arr[j] > insData)
                arr[j + 1] = arr[j];
            else 
                break;
        }
        arr[j + 1] = insData; 
    }
}
```
#### What it does:
- This function implements the **Insertion Sort algorithm** to sort an array in ascending order.

#### Breakdown:
1. **Parameters**:
   - `arr[]`: The array to be sorted.
   - `n`: The number of elements in the array.

2. **Variables**:
   - `insData`: Temporarily stores the value of the current element being inserted.
   - `i` and `j`: Loop counters.

3. **Outer Loop (`for(i = 1; i < n; i++)`)**:
   - Starts at the second element (`i = 1`) and goes to the end of the array.
   - For each element, it treats the left portion of the array as "sorted" and the right portion as "unsorted."

4. **Inner Loop (`for(j = i - 1; j >= 0; j--)`)**:
   - Compares the current element (`insData`) with the elements in the sorted portion.
   - If an element in the sorted portion is greater than `insData`, it shifts that element to the right to make space for `insData`.
   - If an element is smaller or equal, the loop breaks.

5. **Insertion**:
   - After finding the correct position, `insData` is inserted into the array at `arr[j + 1]`.

#### Example:
Let’s say the array is `[5, 2, 4, 6, 1]`:
1. Start with `i = 1` (element `2`):
   - Compare `2` with `5`. Since `5 > 2`, shift `5` to the right.
   - Insert `2` at the first position: `[2, 5, 4, 6, 1]`.
2. Move to `i = 2` (element `4`):
   - Compare `4` with `5`. Since `5 > 4`, shift `5` to the right.
   - Compare `4` with `2`. Since `2 <= 4`, insert `4` at the second position: `[2, 4, 5, 6, 1]`.
3. Repeat until the array is sorted.

#### Why it’s used:
- Insertion Sort is simple and efficient for small datasets or nearly sorted arrays. It sorts the array in place, meaning it doesn’t require additional memory.

---

### **4. PrintArray Function**
```c
static void PrintArray(uint8_t arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
        printf("%d ", arr[i]); 
    }
}
```
#### What it does:
- Prints the contents of the array to the console.

#### Breakdown:
1. **Parameters**:
   - `arr[]`: The array to be printed.

2. **Loop (`for(int i = 0; i < ArrayLength; i++)`)**:
   - Iterates through each element of the array.
   - `printf("%d ", arr[i])` prints the current element followed by a space.

#### Why it’s used:
- This function is used to display the array before and after sorting, so we can see the results.

---

### **5. InputRandomNumber_ToArray Function**
```c
static void InputRandomNumber_ToArray(uint8_t arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
         arr[i] = rand() % 10 + 1;
    }
}
```
#### What it does:
- Fills the array with random numbers between 1 and 10.

#### Breakdown:
1. **Loop (`for(int i = 0; i < ArrayLength; i++)`)**:
   - Iterates through each element of the array.
   - `rand() % 10 + 1` generates a random number between 1 and 10:
     - `rand()` generates a random integer.
     - `% 10` ensures the number is between 0 and 9.
     - `+ 1` shifts the range to 1–10.

#### Why it’s used:
- This function initializes the array with random data, which is necessary for testing the sorting algorithm.

---

### **6. Main Function**
```c
int main(void)
{
    clock_t starttime, endtime;
    starttime = clock();

    srand(time(NULL));   
    uint8_t arr[ArrayLength];

    InputRandomNumber_ToArray(arr);

    printf("Before Insertion Sort\n");
    PrintArray(arr);

    Insertionsort(arr, sizeof(arr) / sizeof(uint8_t)); 
    
    printf("\n\nAfter Insertion Sort\n");
    PrintArray(arr);
   
    printf("\n\n");
    endtime = clock();   
    
    double time_taken = (double)(endtime - starttime) / (double)(CLOCKS_PER_SEC);
    printf("Time taken by program is : %.10f sec\n", time_taken);
    
    return 0; 
}
```
#### What it does:
- The entry point of the program. It orchestrates the entire process:
  1. Measures the start time.
  2. Seeds the random number generator.
  3. Creates and fills the array with random numbers.
  4. Prints the unsorted array.
  5. Sorts the array.
  6. Prints the sorted array.
  7. Measures the end time and calculates the time taken.

#### Breakdown:
1. **Timing**:
   - `clock_t starttime, endtime`: Variables to store the start and end times.
   - `starttime = clock()`: Records the start time.

2. **Random Number Generation**:
   - `srand(time(NULL))`: Seeds the random number generator with the current time.

3. **Array Initialization**:
   - `uint8_t arr[ArrayLength]`: Creates an array of size `ArrayLength`.
   - `InputRandomNumber_ToArray(arr)`: Fills the array with random numbers.

4. **Printing and Sorting**:
   - `PrintArray(arr)`: Prints the unsorted array.
   - `Insertionsort(arr, sizeof(arr) / sizeof(uint8_t))`: Sorts the array.
   - `PrintArray(arr)`: Prints the sorted array.

5. **Timing Calculation**:
   - `endtime = clock()`: Records the end time.
   - `time_taken = (double)(endtime - starttime) / (double)(CLOCKS_PER_SEC)`: Calculates the time taken in seconds.

6. **Output**:
   - Prints the time taken.

#### Why it’s used:
- The `main()` function ties everything together and provides a complete demonstration of the program’s functionality.

---

### **Summary**
This code is a complete program that:
1. Generates an array of random numbers.
2. Sorts the array using Insertion Sort.
3. Prints the array before and after sorting.
4. Measures and displays the time taken for sorting.

Each function has a specific purpose, and the program is structured to make it easy to understand and modify. The use of constants, loops, and functions ensures the code is clean, efficient, and reusable.