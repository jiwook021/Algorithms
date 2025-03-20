# Step-by-Step Explanation: main.c

Let's break down this C code **line by line**, explaining every concept in detail. I'll start from the top and work our way down, explaining each section thoroughly.

---

### **1. Header Files**
```c
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <stdint.h>
```

#### **What it does:**
These lines include external libraries that provide additional functionality to the program.

#### **Explanation:**
- **`#include`**: This is a preprocessor directive that tells the compiler to include the contents of a file (in this case, a library).
- **`<stdio.h>`**: Stands for "Standard Input/Output." It provides functions like `printf` (to print text) and `scanf` (to read input).
- **`<stdlib.h>`**: Stands for "Standard Library." It provides functions like `rand()` (to generate random numbers) and `srand()` (to seed the random number generator).
- **`<time.h>`**: Provides time-related functions, such as `clock()` (to measure time) and `time()` (to get the current time).
- **`<stdint.h>`**: Provides fixed-width integer types, such as `uint16_t` (an unsigned 16-bit integer).

#### **Why it's used:**
These libraries are included because the program needs:
- `printf` to display output
- `rand()` and `srand()` to generate random numbers
- `clock()` to measure execution time
- `uint16_t` to define a fixed-size array length

---

### **2. Global Constant**
```c
const static uint16_t ArrayLength = 100;
```

#### **What it does:**
This line defines a constant value that represents the size of the array.

#### **Explanation:**
- **`const`**: Means the value cannot be changed after it is defined.
- **`static`**: Limits the scope of the variable to this file only (not accessible outside this file).
- **`uint16_t`**: A data type that represents an unsigned 16-bit integer (values from 0 to 65,535).
- **`ArrayLength`**: The name of the constant, set to 100.

#### **Why it's used:**
Using a constant for the array size makes the code more maintainable. If you want to change the array size, you only need to update this one line.

---

### **3. `PrintArray` Function**
```c
static void PrintArray(int arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
        printf("%d ", arr[i]); 
    }
}
```

#### **What it does:**
This function prints all the elements of an array.

#### **Explanation:**
- **`static void`**: 
  - `static` means the function is only visible within this file.
  - `void` means the function does not return a value.
- **`int arr[ArrayLength]`**: The function takes an array of integers as input. The size of the array is `ArrayLength` (100).
- **`for(int i = 0; i < ArrayLength; i++)`**: 
  - A loop that runs 100 times (from `i = 0` to `i = 99`).
  - `i++` increments `i` by 1 after each iteration.
- **`printf("%d ", arr[i]);`**: 
  - Prints the value of `arr[i]` (the `i`-th element of the array).
  - `%d` is a format specifier for integers.

#### **Why it's used:**
This function encapsulates the logic for printing an array, making the code cleaner and reusable.

---

### **4. `InputRandomNumber_ToArray` Function**
```c
static void InputRandomNumber_ToArray(int arr[ArrayLength])
{
    for(int i = 0; i < ArrayLength; i++)
    {
        arr[i] = rand() % 10 + 1;
    }
}
```

#### **What it does:**
This function fills an array with random numbers between 1 and 10.

#### **Explanation:**
- **`rand()`**: Generates a random integer.
- **`rand() % 10 + 1`**: 
  - `rand() % 10` gives a remainder between 0 and 9.
  - Adding 1 shifts the range to 1–10.
- **`arr[i] = rand() % 10 + 1;`**: Stores the random number in the `i`-th position of the array.

#### **Why it's used:**
This function encapsulates the logic for generating random numbers and storing them in the array.

---

### **5. `vBubbleSort` Function**
```c
static void vBubbleSort(int arr[ArrayLength])
{
    int temp;  
    for(int i = 0; i < ArrayLength - 1; i++)
    {
        for(int j = 0; j < ArrayLength - i - 1; j++)
        {
            if (arr[j + 1] < arr[j]) 
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp; 
            }
        }
    }
}
```

#### **What it does:**
This function sorts the array in ascending order using the Bubble Sort algorithm.

#### **Explanation:**
- **Bubble Sort**: A simple sorting algorithm that repeatedly steps through the array, compares adjacent elements, and swaps them if they are in the wrong order.
- **Outer Loop (`for(int i = 0; i < ArrayLength - 1; i++)`)**:
  - Controls the number of passes through the array.
  - After each pass, the largest unsorted element "bubbles up" to its correct position.
- **Inner Loop (`for(int j = 0; j < ArrayLength - i - 1; j++)`)**:
  - Compares adjacent elements and swaps them if necessary.
- **Swapping**:
  - `temp = arr[j];` stores the value of `arr[j]` temporarily.
  - `arr[j] = arr[j + 1];` moves the smaller value to the left.
  - `arr[j + 1] = temp;` moves the larger value to the right.

#### **Why it's used:**
Bubble Sort is used here because it is simple and easy to understand, making it a good choice for educational purposes.

---

### **6. `main` Function**
```c
int main()
{ 
    clock_t starttime, endtime;
    starttime = clock();
    srand(time(NULL));

    int arr[ArrayLength];
    InputRandomNumber_ToArray(arr);
    printf("Before Bubble Sort Array: \n");
    PrintArray(arr);
    vBubbleSort(arr);
    printf("\n\nAfter Bubble Sort Array: \n");
    PrintArray(arr);
    printf("\n");
    
    endtime = clock();
    double time_taken = (double)(endtime - starttime) / (double)(CLOCKS_PER_SEC);
    printf("Time taken by program is : %10f sec\n", time_taken);
    return 0;
}
```

#### **What it does:**
This is the entry point of the program. It:
1. Measures the start time.
2. Seeds the random number generator.
3. Creates an array and fills it with random numbers.
4. Prints the unsorted array.
5. Sorts the array.
6. Prints the sorted array.
7. Measures the end time and calculates the total execution time.

#### **Explanation:**
- **`clock_t`**: A data type used to store time values.
- **`clock()`**: Returns the processor time consumed by the program.
- **`srand(time(NULL));`**: Seeds the random number generator with the current time to ensure different random numbers on each run.
- **`double time_taken = ...`**: Calculates the time taken in seconds by dividing the difference in clock ticks by `CLOCKS_PER_SEC`.

#### **Why it's used:**
The `main` function ties everything together and provides a clear sequence of operations.

---

### **7. Text-Based Diagram of Bubble Sort**
Here’s a simple diagram to illustrate how Bubble Sort works:

```
Initial Array: [5, 3, 8, 4, 6]

Pass 1:
- Compare 5 and 3 → Swap → [3, 5, 8, 4, 6]
- Compare 5 and 8 → No swap
- Compare 8 and 4 → Swap → [3, 5, 4, 8, 6]
- Compare 8 and 6 → Swap → [3, 5, 4, 6, 8]

Pass 2:
- Compare 3 and 5 → No swap
- Compare 5 and 4 → Swap → [3, 4, 5, 6, 8]
- Compare 5 and 6 → No swap

Pass 3:
- Compare 3 and 4 → No swap
- Compare 4 and 5 → No swap

Final Sorted Array: [3, 4, 5, 6, 8]
```

---

This explanation should make the code completely understandable, even for beginners! Let me know if you have further questions.