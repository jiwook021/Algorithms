# Code Overview: main.cpp

This C++ code is a demonstration program that showcases the use of different types of queues and priority queues available in the C++ Standard Template Library (STL). The purpose of the code is to illustrate how to create, manipulate, and display the contents of various queue and priority queue data structures. The code also demonstrates some common operations such as pushing elements, popping elements, and swapping queues.

### Main Functionality and Algorithms Used

1. **Queues (`std::queue`)**:
   - The code demonstrates the use of two types of queues:
     - `queue<int>`: A standard queue that uses a `deque` (double-ended queue) as its underlying container by default.
     - `queue<int, list<int>>`: A queue that uses a `list` as its underlying container.
   - The code generates random numbers and pushes them into these queues. It then prints the contents of the queues.

2. **Priority Queues (`std::priority_queue`)**:
   - The code demonstrates the use of three types of priority queues:
     - `priority_queue<int>`: A standard priority queue that uses a `vector` as its underlying container and `less<int>` as the comparison function, which means it will order elements in descending order (max-heap).
     - `priority_queue<int, vector<int>, greater<int>>`: A priority queue that uses `greater<int>` as the comparison function, which means it will order elements in ascending order (min-heap).
     - `priority_queue<int>` initialized from an array: This shows how to initialize a priority queue with elements from an array.
   - The code generates random numbers and pushes them into these priority queues. It then prints the contents of the priority queues.

3. **Queue and Priority Queue Operations**:
   - The code demonstrates common operations such as:
     - Pushing elements into queues and priority queues.
     - Popping elements from queues and priority queues.
     - Swapping the contents of two priority queues.
     - Printing the contents of queues and priority queues.

4. **Random Number Generation**:
   - The code uses the `rand()` function to generate random numbers, which are then pushed into the queues and priority queues. The `srand()` function is used to seed the random number generator based on the current time.

### Overall Structure

1. **Header Files**:
   - The code includes several standard C++ header files:
     - `<iostream>`: For input/output operations.
     - `<queue>`: For using queues and priority queues.
     - `<list>`: For using the `list` container.
     - `<functional>`: For using comparison functions like `greater<int>`.
     - `<iomanip>`: For formatting output (though it is not used in this code).

2. **Function Definitions**:
   - `print_queue(std::queue<int> q)`: A function to print the contents of a standard queue.
   - `print_queue(std::queue<int, list<int>> q)`: A function to print the contents of a queue that uses a `list` as its underlying container.
   - `print_priority_queue(std::priority_queue<int> q)`: A function to print the contents of a standard priority queue.
   - `print_priority_queue(std::priority_queue<int, vector<int>, greater<int>> q)`: A function to print the contents of a priority queue that uses `greater<int>` as the comparison function.

3. **Main Function**:
   - The `main()` function is where the actual demonstration takes place. It:
     - Initializes random number generation.
     - Creates and manipulates queues and priority queues.
     - Prints the contents of these data structures.
     - Demonstrates swapping of priority queues.
     - Shows how to use `emplace` to add elements to a priority queue.

### Problem Being Solved

The code does not solve a specific real-world problem but serves as an educational example to demonstrate the use of queues and priority queues in C++. It shows how to:
- Create and manipulate different types of queues and priority queues.
- Perform common operations like pushing, popping, and swapping.
- Print the contents of these data structures.

### Approach Taken

The approach taken is to:
1. **Initialize** the necessary data structures (queues and priority queues).
2. **Populate** these data structures with random numbers.
3. **Manipulate** the data structures by performing operations like pushing, popping, and swapping.
4. **Display** the contents of the data structures to show the results of these operations.

### How Different Parts of the Code Work Together

- The **header files** provide the necessary libraries and functions.
- The **function definitions** allow for reusable code to print the contents of queues and priority queues.
- The **main function** ties everything together by creating the data structures, performing operations on them, and using the print functions to display the results.

This code is a comprehensive demonstration of how to work with queues and priority queues in C++, making it a valuable learning tool for understanding these data structures and their operations.