# Step-by-Step Explanation: main.c

Let’s dive into the code step by step, breaking it down in a way that’s accessible to everyone, regardless of their programming experience. I’ll explain each section in detail, define technical terms, and use examples and diagrams to clarify the logic.

---

### **1. The `#include <stdio.h>` Directive**
```c
#include <stdio.h>
```
#### What it does:
This line tells the compiler to include the **Standard Input/Output Library** in the program. This library provides functions like `printf` (used to print text to the console) and `scanf` (used to read input from the user).

#### Why it’s used:
We need `printf` to display the sorted array at the end of the program. Without this library, we wouldn’t be able to print anything to the screen.

---

### **2. The `merge` Function**
```c
void merge(int arr[], int tmp[], int start, int mid, int end)
{
    for(int i = start; i <= end; i++)
    {
        tmp[i] = arr[i];
    }
    int part1 = start;
    int part2 = mid + 1;
    int index = start;
    while(part1 <= mid && part2 <= end)
    {
        if (tmp[part2] > tmp[part1])
        {
           arr[index++] = tmp[part1++];
        }
        else
        {
            arr[index++] = tmp[part2++];
        } 
    }
    while(part1 <= mid)
    {
        arr[index++] = tmp[part1++];   
    }
}
```

#### What it does:
The `merge` function combines two sorted subarrays into a single sorted array. It takes the following parameters:
- `arr[]`: The original array being sorted.
- `tmp[]`: A temporary array used to store a copy of the elements during merging.
- `start`: The starting index of the first subarray.
- `mid`: The ending index of the first subarray (and `mid + 1` is the starting index of the second subarray).
- `end`: The ending index of the second subarray.

#### Step-by-Step Breakdown:
1. **Copy Elements to Temporary Array**:
   ```c
   for(int i = start; i <= end; i++)
   {
       tmp[i] = arr[i];
   }
   ```
   - This loop copies all elements from the original array (`arr`) into the temporary array (`tmp`) for the range `[start, end]`.
   - Example: If `start = 0` and `end = 3`, the loop copies `arr[0]`, `arr[1]`, `arr[2]`, and `arr[3]` into `tmp`.

2. **Initialize Pointers**:
   ```c
   int part1 = start;
   int part2 = mid + 1;
   int index = start;
   ```
   - `part1` points to the start of the first subarray.
   - `part2` points to the start of the second subarray.
   - `index` keeps track of where to place the next element in the original array (`arr`).

3. **Merge the Two Subarrays**:
   ```c
   while(part1 <= mid && part2 <= end)
   {
       if (tmp[part2] > tmp[part1])
       {
          arr[index++] = tmp[part1++];
       }
       else
       {
           arr[index++] = tmp[part2++];
       } 
   }
   ```
   - This loop compares elements from both subarrays (`tmp[part1]` and `tmp[part2]`) and places the smaller element into the original array (`arr`).
   - The `index++` and `part1++`/`part2++` operations ensure that the pointers move forward after each element is placed.
   - Example: If `tmp[part1] = 2` and `tmp[part2] = 3`, the smaller value (`2`) is placed in `arr[index]`, and `part1` is incremented.

4. **Copy Remaining Elements**:
   ```c
   while(part1 <= mid)
   {
       arr[index++] = tmp[part1++];   
   }
   ```
   - If there are any remaining elements in the first subarray (i.e., `part1 <= mid`), they are copied directly into `arr`. This ensures that all elements are included in the final sorted array.

#### Why it’s used:
The `merge` function is the core of the Merge Sort algorithm. It combines two sorted subarrays into a single sorted array, which is essential for the "combine" step of the Divide and Conquer strategy.

---

### **3. The `mergesort` Function**
```c
void mergesort(int arr[], int tmp[], int start, int end)
{
    if(start < end)
    {
        int mid = (start + end) / 2;     
        mergesort(arr, tmp, start, mid);
        mergesort(arr, tmp, mid + 1, end);    
        merge(arr, tmp, start, mid, end);
    }
}
```

#### What it does:
The `mergesort` function recursively divides the array into smaller subarrays, sorts them, and then merges them back together.

#### Step-by-Step Breakdown:
1. **Base Case Check**:
   ```c
   if(start < end)
   ```
   - This condition ensures that the function stops dividing the array when the subarray contains only one element (`start == end`) or is empty (`start > end`).

2. **Divide the Array**:
   ```c
   int mid = (start + end) / 2;
   ```
   - The array is divided into two halves:
     - Left half: `[start, mid]`
     - Right half: `[mid + 1, end]`

3. **Recursive Calls**:
   ```c
   mergesort(arr, tmp, start, mid);
   mergesort(arr, tmp, mid + 1, end);
   ```
   - The function calls itself to sort the left and right halves. This is the **Divide** step of the Divide and Conquer strategy.

4. **Merge the Sorted Halves**:
   ```c
   merge(arr, tmp, start, mid, end);
   ```
   - After the left and right halves are sorted, the `merge` function is called to combine them into a single sorted array.

#### Why it’s used:
The `mergesort` function implements the recursive Divide and Conquer strategy, which is the foundation of the Merge Sort algorithm. It ensures that the array is divided into smaller subarrays, sorted, and then merged back together.

---

### **4. The `Mergesort` Function**
```c
void Mergesort(int arr[])
{ 
    int tmp[8];
    mergesort(arr, tmp, 0, 8);
}
```

#### What it does:
This is a wrapper function that initializes a temporary array and calls the main `mergesort` function with the appropriate parameters.

#### Step-by-Step Breakdown:
1. **Temporary Array**:
   ```c
   int tmp[8];
   ```
   - A temporary array of size 8 is created to store elements during the merging process.

2. **Call `mergesort`**:
   ```c
   mergesort(arr, tmp, 0, 8);
   ```
   - The `mergesort` function is called with the full range of the array (`start = 0`, `end = 8`).

#### Why it’s used:
This function simplifies the interface for sorting the array. Instead of requiring the user to create a temporary array and specify the range, it handles these details internally.

---

### **5. The `main` Function**
```c
int main()
{
    int arr[8] = {10, 2, 3, 4, 5, 6, 7, 3};
    Mergesort(arr);
    for(int i = 0; i < 8; i++)
    {
        printf("%d\n", arr[i]);
    }
}
```

#### What it does:
The `main` function is the entry point of the program. It initializes an array, sorts it using `Mergesort`, and prints the sorted array.

#### Step-by-Step Breakdown:
1. **Initialize the Array**:
   ```c
   int arr[8] = {10, 2, 3, 4, 5, 6, 7, 3};
   ```
   - An array of 8 integers is created with unsorted values.

2. **Sort the Array**:
   ```c
   Mergesort(arr);
   ```
   - The `Mergesort` function is called to sort the array.

3. **Print the Sorted Array**:
   ```c
   for(int i = 0; i < 8; i++)
   {
       printf("%d\n", arr[i]);
   }
   ```
   - A `for` loop is used to print each element of the sorted array.

#### Why it’s used:
The `main` function ties everything together. It initializes the data, calls the sorting function, and displays the results.

---

### **Text-Based Diagram of Merge Sort**
Here’s a simplified diagram of how Merge Sort works on the array `{10, 2, 3, 4, 5, 6, 7, 3}`:

1. **Divide**:
   ```
   {10, 2, 3, 4, 5, 6, 7, 3}
   → {10, 2, 3, 4} and {5, 6, 7, 3}
   → {10, 2} and {3, 4} and {5, 6} and {7, 3}
   → {10} and {2} and {3} and {4} and {5} and {6} and {7} and {3}
   ```

2. **Merge**:
   ```
   {10} and {2} → {2, 10}
   {3} and {4} → {3, 4}
   {5} and {6} → {5, 6}
   {7} and {3} → {3, 7}
   → {2, 10} and {3, 4} → {2, 3, 4, 10}
   → {5, 6} and {3, 7} → {3, 5, 6, 7}
   → {2, 3, 4, 10} and {3, 5, 6, 7} → {2, 3, 3, 4, 5, 6, 7, 10}
   ```

---

### **Summary**
This code implements the Merge Sort algorithm to sort an array of integers. It uses recursion to divide the array into smaller subarrays, sorts them, and then merges them back together. The `merge` function combines two sorted subarrays, the `mergesort` function handles the recursive division and merging, and the `Mergesort` function simplifies the interface. The `main` function initializes the array, calls the sorting function, and prints the sorted result.