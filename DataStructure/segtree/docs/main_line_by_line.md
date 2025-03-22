# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand not just what the code does, but also why it works the way it does.

---

### **1. Global Variables and Includes**
```c++
#include <stdio.h>
#include <stdlib.h>

int segtree [100000];
```

#### **What It Does**
- `#include <stdio.h>` and `#include <stdlib.h>`: These are header files that provide functions for input/output (`printf`, `scanf`) and memory management (`malloc`, `free`), respectively.
- `int segtree[100000]`: This is the array that will store the Segment Tree. It’s large enough to handle up to 100,000 elements.

#### **Why It’s Used**
- The Segment Tree is stored as an array for simplicity and efficiency. Arrays allow fast access to elements using indices, which is crucial for the tree’s operations.

---

### **2. The `update` Function**
```c++
void update(int index, int value, int sz)
{
    int diff = value - segtree[index+sz-1]; 
    for(int i = index+sz-1; i!=0; i/=2)
    {
        segtree[i] += diff;
    }
}
```

#### **What It Does**
This function updates the value of an element in the array and propagates the change through the Segment Tree.

#### **Step-by-Step Breakdown**
1. **Calculate the Difference**:
   - `int diff = value - segtree[index+sz-1];`
     - `index` is the position in the original array.
     - `segtree[index+sz-1]` accesses the leaf node corresponding to the `index`.
     - `diff` is the difference between the new value and the old value.

2. **Propagate the Change**:
   - `for(int i = index+sz-1; i!=0; i/=2)`
     - Start at the leaf node (`index+sz-1`) and move up the tree to the root (`i=0`).
     - At each step, update the current node by adding `diff` to it.
     - `i /= 2` moves to the parent node.

#### **Why It’s Used**
- This ensures that all sums in the tree that depend on the updated value are corrected. The loop runs in **O(log N)** time, making updates efficient.

#### **Example**
Suppose `sz = 8`, `index = 2`, and `value = 5`. If the old value was `3`, `diff = 2`. The loop updates:
- `segtree[9]` (leaf node for index 2)
- `segtree[4]` (parent of node 9)
- `segtree[2]` (parent of node 4)
- `segtree[1]` (root)

---

### **3. The `range` Function**
```c++
int range(int index, int start, int end, int Rangeleft, int RangeRight)
{
    if(start > RangeRight || end < Rangeleft)
        return 0;
    if(Rangeleft <= start && RangeRight >= end)
        return segtree[index];

    int mid = (start + end)/2;
    return range(index*2, start, mid, Rangeleft, RangeRight) + 
           range(index*2+1, mid+1, end, Rangeleft, RangeRight);
}
```

#### **What It Does**
This function computes the sum of elements in the range `[Rangeleft, RangeRight]`.

#### **Step-by-Step Breakdown**
1. **Base Case 1: No Overlap**:
   - `if(start > RangeRight || end < Rangeleft) return 0;`
     - If the current segment (`[start, end]`) doesn’t overlap with the query range, return `0`.

2. **Base Case 2: Full Overlap**:
   - `if(Rangeleft <= start && RangeRight >= end) return segtree[index];`
     - If the current segment is completely inside the query range, return its sum.

3. **Recursive Case: Partial Overlap**:
   - `int mid = (start + end)/2;`
     - Split the current segment into two halves.
   - `return range(index*2, start, mid, Rangeleft, RangeRight) + range(index*2+1, mid+1, end, Rangeleft, RangeRight);`
     - Recursively compute the sum for the left and right halves and add them.

#### **Why It’s Used**
- This recursive approach ensures that only relevant segments are processed, making the query efficient (**O(log N)**).

#### **Example**
Suppose `sz = 8`, `Rangeleft = 2`, `RangeRight = 5`. The function:
1. Starts at the root (`index=1`, `[1, 8]`).
2. Splits into `[1, 4]` and `[5, 8]`.
3. Recursively processes overlapping segments until it finds full overlaps or no overlaps.

---

### **4. The `printtree` Function**
```c++
void printtree(int sz)
{
    printf("\n");
    for(int i = sz*2-1; i>=1; i--)
    {
        printf("%d\n", segtree[i]);    
    }
}
```

#### **What It Does**
This function prints the Segment Tree for debugging or visualization.

#### **Step-by-Step Breakdown**
1. **Loop Through the Tree**:
   - `for(int i = sz*2-1; i>=1; i--)`
     - Start from the last node (`sz*2-1`) and move backward to the root (`i=1`).
   - `printf("%d\n", segtree[i]);`
     - Print the value of each node.

#### **Why It’s Used**
- Visualizing the tree helps debug and understand how updates and queries affect it.

---

### **5. The `main` Function**
```c++
int main()
{
    int n;
    int sz = 1; 
    scanf("%d", &n);
    while(n > sz)
    {
        sz *= 2; 
    }
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &segtree[sz+i]);
    }
    for(int i = sz-1; i >= 1; i--)
    {
        segtree[i] = segtree[i*2] + segtree[i*2+1];   
    }
    printtree(sz);
    update(4, 3, sz);   
    printtree(sz);
    printf("%d", range(1, 1, sz, 2, 3)); 
}
```

#### **What It Does**
This is the driver function that:
1. Reads the input array.
2. Builds the Segment Tree.
3. Demonstrates updates and queries.

#### **Step-by-Step Breakdown**
1. **Determine Tree Size**:
   - `while(n > sz) { sz *= 2; }`
     - Find the smallest power of 2 ≥ `n` to ensure the tree is balanced.

2. **Read Input Values**:
   - `for (int i = 0; i < n; i++) { scanf("%d", &segtree[sz+i]); }`
     - Store input values in the leaves of the tree.

3. **Build the Tree**:
   - `for(int i = sz-1; i >= 1; i--) { segtree[i] = segtree[i*2] + segtree[i*2+1]; }`
     - Compute internal nodes as sums of their children.

4. **Print the Tree**:
   - `printtree(sz);`
     - Visualize the tree after building.

5. **Update and Query**:
   - `update(4, 3, sz);`
     - Update the 4th element to `3`.
   - `printtree(sz);`
     - Visualize the tree after the update.
   - `printf("%d", range(1, 1, sz, 2, 3));`
     - Compute the sum of elements in the range `[2, 3]`.

#### **Why It’s Used**
- This demonstrates the functionality of the Segment Tree in a real-world scenario.

---

### **Text-Based Diagram of the Segment Tree**
Suppose `n = 5` and the input array is `[1, 2, 3, 4, 5]`. The Segment Tree looks like this:

```
        [15]
       /    \
    [10]     [5]
   /   \     /  \
 [3]   [7] [5]  [0]
 / \   / \ / \  / \
[1][2][3][4][5][0][0][0]
```

- The root (`segtree[1]`) stores the sum of the entire array (`15`).
- Internal nodes store sums of their children.
- Leaves store the original array values.

---

### **Summary**
This code implements a Segment Tree for efficient range sum queries and point updates. It uses:
- **Bottom-up construction** to build the tree.
- **Recursive queries** to compute range sums.
- **Efficient updates** to propagate changes.

Let me know if you’d like further clarification or improvements!