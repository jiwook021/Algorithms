# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, even if you’re a beginner.

---

### **1. Header Files and Constants**
```cpp
#include <iostream>
#include <stdlib.h> 
#include <time.h>
#include <iomanip> 

static const uint8_t ArrayLength = 100;
```

#### **What It Does**
- These lines include necessary libraries and define a constant for the array size.

#### **Explanation**
1. **Header Files**:
   - `#include <iostream>`: Provides input/output functionality (e.g., `std::cout` for printing to the console).
   - `#include <stdlib.h>`: Includes functions like `rand()` (for random number generation) and `malloc()` (for dynamic memory allocation).
   - `#include <time.h>`: Provides timing functions like `clock()` and `time()`.
   - `#include <iomanip>`: Allows formatting of output (e.g., setting decimal precision).

2. **Constant Definition**:
   - `static const uint8_t ArrayLength = 100;`: Defines a constant named `ArrayLength` with a value of 100. This determines the size of the array used in the program.
     - `uint8_t`: A data type that represents an unsigned 8-bit integer (values from 0 to 255).
     - `static const`: Ensures the value cannot be changed and is only visible within this file.

#### **Why It’s Used**
- Constants like `ArrayLength` make the code more readable and maintainable. If you need to change the array size, you only need to update this one line.

---

### **2. `PrintArray` Function**
```cpp
static void PrintArray(int arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
        std::cout << arr[i] << " "; 
    }
}
```

#### **What It Does**
- This function prints all the elements of an array to the console.

#### **Explanation**
1. **Function Signature**:
   - `static void PrintArray(int arr[ArrayLength])`: Defines a function named `PrintArray` that takes an array of integers as input.
     - `static`: Limits the function’s visibility to this file.
     - `void`: Indicates the function does not return a value.
     - `int arr[ArrayLength]`: The function accepts an array of size `ArrayLength`.

2. **For Loop**:
   - `for(int i = 0; i < ArrayLength; i++)`: Iterates over each element of the array.
     - `i`: A loop counter that starts at 0 and increments by 1 each iteration.
     - `i < ArrayLength`: The loop continues as long as `i` is less than the array size.
     - `i++`: Increments `i` after each iteration.

3. **Printing**:
   - `std::cout << arr[i] << " ";`: Prints the current array element followed by a space.

#### **Why It’s Used**
- This function is a utility to display the contents of the array, making it easier to verify the program’s correctness.

---

### **3. `InputRandomNumber_ToArray` Function**
```cpp
static void InputRandomNumber_ToArray(int arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
        arr[i] = rand() % 10 + 1;
    }
}
```

#### **What It Does**
- Fills an array with random integers between 1 and 10.

#### **Explanation**
1. **Function Signature**:
   - Similar to `PrintArray`, this function takes an array as input.

2. **For Loop**:
   - Iterates over each element of the array.

3. **Random Number Generation**:
   - `arr[i] = rand() % 10 + 1;`:
     - `rand()`: Generates a random integer.
     - `% 10`: Limits the range to 0–9.
     - `+ 1`: Shifts the range to 1–10.

#### **Why It’s Used**
- Random numbers are used to simulate real-world data and test the sorting algorithm.

---

### **4. `MergeTwoArea` Function**
```cpp
static void MergeTwoArea(int arr[], int left, int mid, int right)
{
    int fIdx = left;
    int rIdx = mid + 1; 
    int i;

    int *sortArr = (int*)malloc(sizeof(int) * (right + 1));
    int sIdx = left; 

    while(fIdx <= mid && rIdx <= right)
    {    
        if(arr[fIdx] <= arr[rIdx])
            sortArr[sIdx] = arr[fIdx++];
        else 
            sortArr[sIdx] = arr[rIdx++]; 

        sIdx++; 
    }

    if(fIdx > mid)
    {
        for(i = rIdx; i <= right; i++, sIdx++)
            sortArr[sIdx] = arr[i];
    }
    else 
    {
        for(i = fIdx; i <= mid; i++, sIdx++)
            sortArr[sIdx] = arr[i];
    }

    for(i = left; i <= right; i++)
    {
        arr[i] = sortArr[i];
    }

    free(sortArr);
}
```

#### **What It Does**
- Merges two sorted subarrays into a single sorted array.

#### **Explanation**
1. **Parameters**:
   - `arr[]`: The array to be sorted.
   - `left`: The starting index of the first subarray.
   - `mid`: The ending index of the first subarray.
   - `right`: The ending index of the second subarray.

2. **Temporary Array**:
   - `int *sortArr = (int*)malloc(sizeof(int) * (right + 1));`: Allocates memory for a temporary array to store the merged result.

3. **Merging Logic**:
   - The `while` loop compares elements from both subarrays and copies the smaller one to `sortArr`.
   - The `if` and `else` blocks handle any remaining elements in either subarray.

4. **Copy Back**:
   - The final `for` loop copies the sorted elements from `sortArr` back to the original array.

5. **Memory Cleanup**:
   - `free(sortArr);`: Releases the allocated memory.

#### **Why It’s Used**
- This function is a key part of the Merge Sort algorithm, combining two sorted subarrays into one.

---

### **5. `MergeSort` Function**
```cpp
static void MergeSort(int arr[], int left, int right)
{
    int mid; 
    if(left < right) 
    {
        mid = (left + right) / 2; 
        MergeSort(arr, left, mid);
        MergeSort(arr, mid + 1, right);
        MergeTwoArea(arr, left, mid, right);   
    }
}
```

#### **What It Does**
- Recursively sorts an array using the Merge Sort algorithm.

#### **Explanation**
1. **Base Case**:
   - `if(left < right)`: Ensures the function stops when the subarray has one element.

2. **Divide**:
   - `mid = (left + right) / 2;`: Splits the array into two halves.

3. **Conquer**:
   - `MergeSort(arr, left, mid);`: Recursively sorts the left half.
   - `MergeSort(arr, mid + 1, right);`: Recursively sorts the right half.

4. **Combine**:
   - `MergeTwoArea(arr, left, mid, right);`: Merges the two sorted halves.

#### **Why It’s Used**
- This function implements the divide-and-conquer strategy of Merge Sort.

---

### **6. `main` Function**
```cpp
int main(void)
{
    clock_t starttime, endtime;
    starttime = clock();
    srand(time(NULL));

    int arr[ArrayLength];

    InputRandomNumber_ToArray(arr);
    printf("\nBefore Merge Sort: ");
    PrintArray(arr);   
    MergeSort(arr, 0, sizeof(arr) / sizeof(int) - 1); 
    printf("\nAfter Merge Sort: ");
    PrintArray(arr);   
    printf("\n\n");

    endtime = clock();   
    double time_taken = double(endtime - starttime) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken by program is : " << std::fixed << time_taken << std::setprecision(10);
    std::cout << " sec " << std::endl;
    return 0; 
}
```

#### **What It Does**
- Coordinates the program’s execution, including array generation, sorting, and performance measurement.

#### **Explanation**
1. **Timing**:
   - `clock_t starttime, endtime;`: Variables to store start and end times.
   - `starttime = clock();`: Records the start time.

2. **Random Seed**:
   - `srand(time(NULL));`: Seeds the random number generator.

3. **Array Initialization**:
   - `int arr[ArrayLength];`: Declares an array of size `ArrayLength`.

4. **Array Population**:
   - `InputRandomNumber_ToArray(arr);`: Fills the array with random numbers.

5. **Pre-Sort Output**:
   - `PrintArray(arr);`: Displays the unsorted array.

6. **Sorting**:
   - `MergeSort(arr, 0, sizeof(arr) / sizeof(int) - 1);`: Sorts the array.

7. **Post-Sort Output**:
   - `PrintArray(arr);`: Displays the sorted array.

8. **Performance Measurement**:
   - `endtime = clock();`: Records the end time.
   - `double time_taken = double(endtime - starttime) / double(CLOCKS_PER_SEC);`: Calculates the elapsed time in seconds.

9. **Output**:
   - Prints the time taken to sort the array.

#### **Why It’s Used**
- The `main` function ties everything together, demonstrating the entire process from array generation to sorting and performance measurement.

---

### **Summary**
This code is a complete implementation of the Merge Sort algorithm, with additional features for random array generation, output display, and performance measurement. Each function has a specific role, and the program as a whole demonstrates how to structure and analyze a sorting algorithm in C++.