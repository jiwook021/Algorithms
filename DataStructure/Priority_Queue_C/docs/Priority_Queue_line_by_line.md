# Step-by-Step Explanation: Priority_Queue.c

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Header Files and Definitions**
```c
#include "Priority_Queue.h"
#include <stdbool.h>
#include <stdio.h>
```
- **What it does**: These lines include necessary header files.
  - `Priority_Queue.h`: Likely contains the definition of the `heap` structure and other related types.
  - `stdbool.h`: Provides the `bool` type and `true`/`false` constants, used for boolean logic.
  - `stdio.h`: Provides input/output functions like `printf`.

- **Why it’s used**: Header files allow the program to use predefined structures, types, and functions. For example, `stdbool.h` is used to make the code more readable by using `true` and `false` instead of `1` and `0`.

---

### **2. Heap Initialization**
```c
void heapInit(heap* ph)
{
    ph->numOfData = 0; 
}
```
- **What it does**: This function initializes a heap by setting the number of elements (`numOfData`) to 0.
- **Why it’s used**: Before using the heap, it must be initialized to an empty state. This ensures that the heap starts with no elements.
- **Example**: If you create a new heap, `heapInit` sets it up so that it’s ready to accept new elements.

---

### **3. Check if Heap is Empty**
```c
int HIsEmpty(heap* ph)
{
    if (ph->numOfData == 0)
        return true;
    else
        return false;
}
```
- **What it does**: This function checks if the heap is empty by verifying if `numOfData` is 0.
- **Why it’s used**: It’s a utility function to avoid errors when trying to delete from an empty heap.
- **Example**: Before calling `Hdelete`, you can use `HIsEmpty` to ensure the heap isn’t empty.

---

### **4. Helper Functions for Heap Navigation**
```c
int GetParentIDX(int idx)
{
    return idx / 2;
}

int GetLChildIDX(int idx)
{
    return idx * 2;
}

int GetRChildIDX(int idx)
{
    return GetLChildIDX(idx) + 1; 
}
```
- **What it does**: These functions calculate the indices of a node’s parent, left child, and right child in the heap array.
  - `GetParentIDX`: Returns the index of the parent of the node at index `idx`.
  - `GetLChildIDX`: Returns the index of the left child of the node at index `idx`.
  - `GetRChildIDX`: Returns the index of the right child of the node at index `idx`.

- **Why it’s used**: In a binary heap, the parent-child relationships are determined by their positions in the array. These functions make it easy to navigate the heap.
- **Example**: For a node at index `3`:
  - Parent: `GetParentIDX(3)` returns `1`.
  - Left child: `GetLChildIDX(3)` returns `6`.
  - Right child: `GetRChildIDX(3)` returns `7`.

---

### **5. Insertion into the Heap**
```c
void HInsert(heap* ph, HData data, Priority pr)
{
    int idx = ph->numOfData + 1; 
    HeapElem nelem = { pr, data };

    while (idx != 1)
    {
        if (pr < (ph->heapArr[GetParentIDX(idx)].pr))
        {
            ph->heapArr[idx] = ph->heapArr[GetParentIDX(idx)];
            idx = GetParentIDX(idx); 
        }
        else 
            break;
    } 
    ph->heapArr[idx] = nelem; 
    ph->numOfData += 1; 
    printf("\nInserted %d with priority of %d ", (nelem.data), (nelem.pr));
}
```
- **What it does**: This function inserts a new element into the heap while maintaining the heap property.
- **Step-by-step breakdown**:
  1. **Calculate the insertion index**: The new element is added at the next available position (`ph->numOfData + 1`).
  2. **Create a new element**: `nelem` is a structure containing the data and its priority.
  3. **Restore the heap property**:
     - Compare the new element’s priority with its parent’s priority.
     - If the new element has a higher priority (lower value), swap it with its parent.
     - Repeat this process until the heap property is satisfied or the root is reached.
  4. **Update the heap**: Place the new element in its correct position and increment the number of elements.

- **Why it’s used**: This ensures that the heap remains a valid min-heap after insertion.
- **Example**: Inserting an element with priority `2`:
  - If the parent has priority `3`, the new element is swapped with the parent.
  - If the parent has priority `1`, the loop breaks, and the new element is placed in its current position.

---

### **6. Finding the Higher-Priority Child**
```c
int getHiPriChiledIDX(heap* ph, int idx)
{
    if (GetLChildIDX(idx) > ph->numOfData)
        return 0;
    else if (GetLChildIDX(idx) == ph->numOfData)
        return GetLChildIDX(idx);
    else
    {
        if (ph->heapArr[GetLChildIDX(idx)].pr > ph->heapArr[GetRChildIDX(idx)].pr)
            return GetRChildIDX(idx);
        else
            return GetLChildIDX(idx);
    }
}
```
- **What it does**: This function determines which child of a node has the higher priority.
- **Step-by-step breakdown**:
  1. **Check if the left child exists**: If the left child’s index is beyond the heap size, return `0` (no child).
  2. **Check if only the left child exists**: If the left child is the last element, return its index.
  3. **Compare priorities**: If both children exist, return the index of the child with the higher priority (lower value).

- **Why it’s used**: During deletion, this function helps determine which child should replace the parent to maintain the heap property.
- **Example**: For a node at index `2`:
  - If the left child has priority `3` and the right child has priority `2`, the function returns the index of the right child.

---

### **7. Deletion from the Heap**
```c
HData Hdelete(heap* ph)
{
    HData retData = (ph->heapArr[1]).data; 
    HeapElem lastElem = ph->heapArr[ph->numOfData]; 

    int parentIdx = 1; 
    int childIdx; 

    while (childIdx = getHiPriChiledIDX(ph, parentIdx))
    {
        if (lastElem.pr <= ph->heapArr[childIdx].pr)
            break;
        ph->heapArr[parentIdx] = ph->heapArr[childIdx];
        parentIdx = childIdx;
    }
    ph->heapArr[parentIdx] = lastElem;
    ph->numOfData -= 1; 
    return retData; 
}
```
- **What it does**: This function removes and returns the element with the highest priority (the root) while maintaining the heap property.
- **Step-by-step breakdown**:
  1. **Save the root element**: The root (highest priority element) is stored in `retData`.
  2. **Move the last element to the root**: The last element in the heap is moved to the root position.
  3. **Restore the heap property**:
     - Compare the new root with its children.
     - If the new root has a lower priority (higher value), swap it with the child of higher priority.
     - Repeat this process until the heap property is satisfied.
  4. **Update the heap**: Place the last element in its correct position and decrement the number of elements.

- **Why it’s used**: This ensures that the heap remains a valid min-heap after deletion.
- **Example**: Deleting the root:
  - If the root has priority `1`, it is removed, and the last element (e.g., priority `5`) is moved to the root.
  - The new root is compared with its children and swapped until the heap property is restored.

---

### **8. Summary**
This code implements a priority queue using a binary heap. The heap is represented as an array, and the heap property is maintained using helper functions to navigate the tree structure. The key operations are:
- **Insertion**: Adds a new element and restores the heap property by "bubbling up."
- **Deletion**: Removes the highest-priority element and restores the heap property by "bubbling down."

Let me know if you’d like further clarification or improvements!