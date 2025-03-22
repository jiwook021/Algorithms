# Step-by-Step Explanation: main.c

Let’s break down the code **step by step** in a way that’s easy to understand, even for someone who is just starting to learn programming. I’ll explain every significant section, define technical terms, and use examples and diagrams to make everything clear.

---

### **1. Header Files**
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
```

#### What It Does:
These lines include standard libraries that provide essential functions and tools for the program:
- `stdio.h`: Provides input/output functions like `printf`.
- `stdlib.h`: Provides memory management functions like `malloc` and `free`.
- `stdbool.h`: Allows the use of `bool` (boolean) type for true/false values.

#### Why It’s Used:
- Without these libraries, we wouldn’t be able to print output, allocate memory, or use boolean values.

---

### **2. Defining the `node` Structure**
```c
typedef struct Node
{
    int data;
    struct Node* prev;
} node;
```

#### What It Does:
- Defines a `node` structure, which represents a single element in the stack.
- Each `node` contains:
  - `data`: An integer value stored in the node.
  - `prev`: A pointer to the previous node in the stack.

#### Why It’s Used:
- A **linked list** is used to implement the stack. Each node points to the previous node, creating a chain of elements.
- This allows the stack to grow and shrink dynamically, unlike arrays, which have a fixed size.

#### Example:
If the stack contains the values `[10, 20, 30]`, the nodes would look like this:
```
top -> [30] -> [20] -> [10] -> NULL
```
- The `prev` pointer of the `30` node points to the `20` node, and so on.

---

### **3. Defining the `stack` Structure**
```c
typedef struct Stack
{
    node* top;
    int sz;
} stack;
```

#### What It Does:
- Defines a `stack` structure, which represents the stack itself.
- It contains:
  - `top`: A pointer to the top node in the stack.
  - `sz`: The size of the stack (number of elements).

#### Why It’s Used:
- The `top` pointer allows quick access to the most recently added element.
- The `sz` variable keeps track of the stack’s size, making it easy to check if the stack is empty.

---

### **4. `initstack` Function**
```c
stack* initstack()
{
    stack* s1 = (stack*)malloc(sizeof(stack));
    s1->top = NULL;
    s1->sz = 0;
    return s1;
}
```

#### What It Does:
- Creates and initializes a new stack.
- Allocates memory for the stack structure using `malloc`.
- Sets the `top` pointer to `NULL` (indicating an empty stack) and `sz` to `0`.

#### Why It’s Used:
- `malloc` dynamically allocates memory, allowing the stack to exist in memory until explicitly freed.
- Initializing `top` to `NULL` and `sz` to `0` ensures the stack starts in a valid, empty state.

#### Example:
```c
stack* myStack = initstack();
```
- This creates an empty stack with `top = NULL` and `sz = 0`.

---

### **5. `empty` Function**
```c
bool empty(stack* s)
{
    return (s->sz) ? 0 : 1;
}
```

#### What It Does:
- Checks if the stack is empty.
- Returns `1` (true) if the stack size (`sz`) is `0`, otherwise returns `0` (false).

#### Why It’s Used:
- Provides a simple way to check if the stack has any elements before performing operations like `pop`.

#### Example:
```c
if (empty(myStack)) {
    printf("Stack is empty!\n");
}
```

---

### **6. `push` Function**
```c
void push(stack* s, int data)
{
    node* mynode = (node*)malloc(sizeof(node));
    mynode->data = data;

    if (empty(s))
    {
        s->top = mynode;
    }
    mynode->prev = s->top;
    s->top = mynode;
    s->sz++;
}
```

#### What It Does:
- Adds a new element to the top of the stack.
- Creates a new node, sets its `data`, and links it to the stack.

#### Step-by-Step Logic:
1. Allocate memory for a new node using `malloc`.
2. Set the `data` field of the node to the value being pushed.
3. If the stack is empty (`empty(s)` is true), set the `top` pointer to the new node.
4. Link the new node to the current `top` node by setting `mynode->prev = s->top`.
5. Update the `top` pointer to point to the new node.
6. Increment the stack size (`sz`).

#### Why It’s Used:
- This function implements the **LIFO (Last-In-First-Out)** principle of stacks.
- The new node becomes the new `top`, and its `prev` pointer links to the previous `top`.

#### Example:
```c
push(myStack, 10);
push(myStack, 20);
```
- After these operations, the stack looks like:
```
top -> [20] -> [10] -> NULL
```

---

### **7. `pop` Function**
```c
int pop(stack* s)
{
    if (empty(s))
        return -1;
    node* temp = s->top;
    int data = temp->data;
    s->top = s->top->prev;
    free(temp);
    s->sz--;
    return data;
}
```

#### What It Does:
- Removes and returns the top element from the stack.
- If the stack is empty, it returns `-1` to indicate an error.

#### Step-by-Step Logic:
1. Check if the stack is empty using `empty(s)`. If true, return `-1`.
2. Store the current `top` node in a temporary pointer (`temp`).
3. Save the `data` from the `top` node.
4. Update the `top` pointer to point to the previous node (`s->top->prev`).
5. Free the memory of the removed node using `free(temp)`.
6. Decrement the stack size (`sz`).
7. Return the saved `data`.

#### Why It’s Used:
- Implements the **LIFO** principle by removing the most recently added element.
- Ensures memory is properly freed to avoid leaks.

#### Example:
```c
int value = pop(myStack);
```
- If the stack is `[20] -> [10] -> NULL`, `value` will be `20`, and the stack becomes `[10] -> NULL`.

---

### **8. `main` Function**
```c
int main()
{
    stack* s1 = initstack();

    push(s1, 10);
    printf("hi\n");
    printf("%d\n", pop(s1));

    for (int i = 0; i < 10; i++)
        push(s1, i);

    for (int i = 0; i < 11; i++)
        printf("%d\n", pop(s1));

    return 0;
}
```

#### What It Does:
- Tests the stack implementation by performing a series of `push` and `pop` operations.

#### Step-by-Step Logic:
1. Create a new stack using `initstack()`.
2. Push the value `10` onto the stack.
3. Print "hi" to the console.
4. Pop the top value (`10`) and print it.
5. Use a loop to push values `0` through `9` onto the stack.
6. Use another loop to pop and print all values, including an extra pop to test handling an empty stack.

#### Why It’s Used:
- Demonstrates the functionality of the stack.
- Tests edge cases, such as popping from an empty stack.

#### Example Output:
```
hi
10
9
8
7
6
5
4
3
2
1
0
-1
```
- The `-1` indicates an attempt to pop from an empty stack.

---

### **Text-Based Diagram of Stack Operations**

#### Initial State:
```
top -> NULL
sz = 0
```

#### After `push(s1, 10)`:
```
top -> [10] -> NULL
sz = 1
```

#### After `push(s1, 20)`:
```
top -> [20] -> [10] -> NULL
sz = 2
```

#### After `pop(s1)`:
```
top -> [10] -> NULL
sz = 1
```

---

This explanation should make the code completely understandable, even for beginners. Let me know if you’d like to dive deeper into any specific part!