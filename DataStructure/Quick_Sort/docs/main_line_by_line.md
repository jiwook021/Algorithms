# Step-by-Step Explanation: main.c

Absolutely! Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll have a deep understanding of how this code works, even if you’re new to programming.

---

### **1. The `Swap` Function**
```c
void Swap(int arr[], int idx1, int idx2)
{
    int temp = arr[idx1];
    arr[idx1] = arr[idx2];
    arr[idx2] = temp;
}
```

#### **What It Does**
This function swaps two elements in an array. It takes three inputs:
1. `arr[]`: The array where the elements are stored.
2. `idx1`: The index of the first element to swap.
3. `idx2`: The index of the second element to swap.

#### **How It Works**
1. **Store the First Element**:
   - `int temp = arr[idx1];`
   - The value at `arr[idx1]` is temporarily stored in a variable called `temp`.

2. **Replace the First Element**:
   - `arr[idx1] = arr[idx2];`
   - The value at `arr[idx2]` is copied to `arr[idx1]`.

3. **Replace the Second Element**:
   - `arr[idx2] = temp;`
   - The value stored in `temp` (the original value of `arr[idx1]`) is copied to `arr[idx2]`.

#### **Why It’s Used**
Swapping is a fundamental operation in sorting algorithms. It allows us to rearrange elements in the array so that smaller elements move to the left and larger elements move to the right.

#### **Example**
If `arr = [3, 2, 4]`, and we call `Swap(arr, 0, 1)`:
1. `temp = arr[0] = 3`
2. `arr[0] = arr[1] = 2`
3. `arr[1] = temp = 3`
The array becomes `[2, 3, 4]`.

---

### **2. The `Partition` Function**
```c
int Partition(int arr[], int left, int right)
{
    int pivot = arr[left];
    int low = left + 1;
    int high = right;

    while (low <= high)
    {
        while (pivot > arr[low])
            low++;
        while (pivot < arr[high])
            high--;

        if (low <= high)
            Swap(arr, low, high);
    }

    Swap(arr, left, high);
    return high;
}
```

#### **What It Does**
This function partitions the array into two parts:
1. Elements smaller than the pivot.
2. Elements larger than the pivot.

It returns the index of the pivot after partitioning.

#### **How It Works**
1. **Choose the Pivot**:
   - `int pivot = arr[left];`
   - The pivot is the first element in the current sub-array.

2. **Initialize Pointers**:
   - `int low = left + 1;`
     - `low` starts just after the pivot and moves right.
   - `int high = right;`
     - `high` starts at the end of the sub-array and moves left.

3. **Move Pointers**:
   - The `while (low <= high)` loop continues until `low` and `high` cross each other.
   - Inside the loop:
     - `while (pivot > arr[low]) low++;`
       - Move `low` to the right until an element greater than the pivot is found.
     - `while (pivot < arr[high]) high--;`
       - Move `high` to the left until an element smaller than the pivot is found.

4. **Swap Elements**:
   - If `low` and `high` haven’t crossed (`if (low <= high)`), swap the elements at `low` and `high`.

5. **Place the Pivot**:
   - After the loop, swap the pivot (`arr[left]`) with `arr[high]`.
   - This places the pivot in its correct position.

6. **Return the Pivot Index**:
   - The function returns `high`, which is the index of the pivot.

#### **Why It’s Used**
Partitioning is the core of Quicksort. It ensures that elements smaller than the pivot are on the left, and elements larger than the pivot are on the right. This allows the algorithm to sort the array recursively.

#### **Example**
Let’s partition the array `[3, 2, 4, 1, 7, 6, 5]`:
1. Pivot: `3`
2. `low` starts at `2` (index 1), `high` starts at `5` (index 6).
3. Move `low` to `4` (index 2), `high` to `1` (index 3).
4. Swap `4` and `1`: `[3, 2, 1, 4, 7, 6, 5]`.
5. Move `low` to `4` (index 3), `high` to `1` (index 2).
6. Swap pivot (`3`) with `1`: `[1, 2, 3, 4, 7, 6, 5]`.
7. Return `high = 2` (index of pivot `3`).

---

### **3. The `QuickSort` Function**
```c
void QuickSort(int arr[], int left, int right)
{
    if (left <= right)
    {
        int pivot = Partition(arr, left, right);
        QuickSort(arr, left, pivot - 1);
        QuickSort(arr, pivot + 1, right);
    }
}
```

#### **What It Does**
This function recursively sorts the array using the Quicksort algorithm.

#### **How It Works**
1. **Base Case**:
   - If `left <= right`, the function proceeds. Otherwise, it stops (no elements to sort).

2. **Partition the Array**:
   - `int pivot = Partition(arr, left, right);`
   - The array is partitioned, and the pivot index is returned.

3. **Recursively Sort Sub-arrays**:
   - `QuickSort(arr, left, pivot - 1);`
     - Sort the left sub-array (elements smaller than the pivot).
   - `QuickSort(arr, pivot + 1, right);`
     - Sort the right sub-array (elements larger than the pivot).

#### **Why It’s Used**
Quicksort is a divide-and-conquer algorithm. By recursively sorting smaller sub-arrays, it efficiently sorts the entire array.

#### **Example**
For the array `[3, 2, 4, 1, 7, 6, 5]`:
1. Partition: `[1, 2, 3, 4, 7, 6, 5]` (pivot at index 2).
2. Sort left sub-array `[1, 2]`:
   - Partition: `[1, 2]` (pivot at index 0).
   - Sort left sub-array `[]` (empty).
   - Sort right sub-array `[2]` (already sorted).
3. Sort right sub-array `[4, 7, 6, 5]`:
   - Partition: `[4, 5, 6, 7]` (pivot at index 3).
   - Sort left sub-array `[4, 5, 6]`.
   - Sort right sub-array `[]` (empty).

---

### **4. The `main` Function**
```c
int main()
{
    int arr[7] = {3, 2, 4, 1, 7, 6, 5};
    int len = sizeof(arr) / sizeof(int);

    printf("Before Quicksort: ");
    int i;
    for (i = 0; i < len; i++)
        printf("%d", arr[i]);

    QuickSort(arr, 0, len - 1);

    printf("\n\nAfter Quicksort: ");
    for (i = 0; i < len; i++)
        printf("%d", arr[i]);

    printf("\n");
    return 0;
}
```

#### **What It Does**
This is the entry point of the program. It:
1. Initializes an array.
2. Prints the unsorted array.
3. Sorts the array using Quicksort.
4. Prints the sorted array.

#### **How It Works**
1. **Initialize the Array**:
   - `int arr[7] = {3, 2, 4, 1, 7, 6, 5};`
   - Creates an array with 7 elements.

2. **Calculate Array Length**:
   - `int len = sizeof(arr) / sizeof(int);`
   - `sizeof(arr)` gives the total size of the array in bytes.
   - `sizeof(int)` gives the size of one integer in bytes.
   - Dividing them gives the number of elements in the array.

3. **Print the Unsorted Array**:
   - A `for` loop iterates through the array and prints each element.

4. **Sort the Array**:
   - `QuickSort(arr, 0, len - 1);`
   - Calls the `QuickSort` function to sort the array.

5. **Print the Sorted Array**:
   - Another `for` loop prints the sorted array.

6. **Return 0**:
   - Indicates that the program executed successfully.

---

### **Summary**
This code demonstrates how to implement Quicksort in C. It uses recursion, partitioning, and swapping to efficiently sort an array. Each function has a clear purpose, and the code is modular and easy to understand. Let me know if you’d like to explore potential improvements next!