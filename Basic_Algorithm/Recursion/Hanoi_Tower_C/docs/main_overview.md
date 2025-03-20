# Code Overview: main.c

### Purpose of the Code

This C code is designed to solve the **Tower of Hanoi** problem, a classic mathematical puzzle and a common example used in computer science to demonstrate recursion. The Tower of Hanoi problem involves moving a stack of disks from one peg to another, following specific rules:

1. **Only one disk can be moved at a time.**
2. **A disk can only be placed on top of a larger disk or on an empty peg.**
3. **The goal is to move the entire stack from the starting peg to the destination peg, using a middle peg as auxiliary.**

The code uses a **recursive algorithm** to solve this problem. Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. In this case, the `hanoi` function is called recursively to move disks between the pegs.

### Main Functionality

The code consists of two main parts:

1. **The `hanoi` function**: This is the core of the program. It takes four parameters:
   - `N`: The number of disks to move.
   - `src`: The source peg (where the disks start).
   - `middle`: The auxiliary or middle peg (used for temporary storage).
   - `dest`: The destination peg (where the disks need to end up).

   The function uses recursion to move `N` disks from the source peg to the destination peg, following the rules of the Tower of Hanoi.

2. **The `main` function**: This is the entry point of the program. It calls the `hanoi` function with specific parameters to solve the problem for 2 disks, moving them from peg 'A' to peg 'C', using peg 'B' as the auxiliary.

### Algorithm Used

The algorithm used in this code is a **recursive divide-and-conquer approach**. Here's how it works:

1. **Base Case**: If there's only one disk (`N == 1`), move it directly from the source peg to the destination peg. This is the simplest case and doesn't require any further recursion.

2. **Recursive Case**: If there are more than one disks (`N > 1`), the problem is broken down into three steps:
   - **Step 1**: Move the top `N-1` disks from the source peg to the middle peg, using the destination peg as auxiliary.
   - **Step 2**: Move the largest disk (the `N`th disk) from the source peg to the destination peg.
   - **Step 3**: Move the `N-1` disks from the middle peg to the destination peg, using the source peg as auxiliary.

   This process is repeated recursively until all disks are moved to the destination peg.

### Overall Structure

The code is structured as follows:

1. **Function Prototype**: The `hanoi` function is declared at the top of the file. This is a good practice in C to ensure that the function is known before it is used.

2. **Function Definition**: The `hanoi` function is defined with the logic to move the disks. It uses recursion to solve the problem.

3. **Main Function**: The `main` function is the entry point of the program. It calls the `hanoi` function with the initial parameters to start the disk-moving process.

### How the Parts Work Together

- The `main` function calls the `hanoi` function with the initial parameters (`N=2`, `src='A'`, `middle='B'`, `dest='C'`).
- The `hanoi` function checks if `N` is 1. If so, it moves the disk directly from the source to the destination.
- If `N` is greater than 1, the function calls itself recursively to move `N-1` disks from the source to the middle peg, then moves the `N`th disk from the source to the destination, and finally moves the `N-1` disks from the middle peg to the destination.
- This recursive process continues until all disks are moved to the destination peg.

### Example Execution

For `N=2`, the execution would look like this:

1. **First Call**: `hanoi(2, 'A', 'B', 'C')`
   - Move 1 disk from 'A' to 'B' (using 'C' as auxiliary).
   - Move the 2nd disk from 'A' to 'C'.
   - Move the 1 disk from 'B' to 'C' (using 'A' as auxiliary).

2. **Output**:
   - `moved 1 from src A to dest B`
   - `moved 2 from src A to dest C`
   - `moved 1 from src B to dest C`

This output shows the sequence of moves required to solve the Tower of Hanoi problem for 2 disks.

### Summary

The code is a classic example of using recursion to solve the Tower of Hanoi problem. It demonstrates how a complex problem can be broken down into simpler subproblems, each of which is solved recursively. The `hanoi` function is the heart of the program, and the `main` function sets up the initial conditions for the problem. Together, they provide a clear and concise solution to the Tower of Hanoi puzzle.