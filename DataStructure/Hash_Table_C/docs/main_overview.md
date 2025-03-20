# Code Overview: main.c

This C code demonstrates the implementation and usage of a **hash table** to store and manage person-related data. Let's break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The code is designed to:
1. **Store and manage person data** (e.g., Social Security Numbers (SSNs), names, and addresses) in a hash table.
2. **Insert**, **search**, and **delete** person records efficiently using the hash table data structure.
3. Demonstrate how a hash table can be used to handle collisions and retrieve data quickly.

The hash table is a data structure that maps keys (in this case, SSNs) to values (person records) using a hash function. This allows for fast insertion, lookup, and deletion of data.

---

### **Main Functionality**
1. **Hash Table Initialization**:
   - The hash table is initialized with a custom hash function (`MyHashFunc`), which determines where each person's data will be stored in the table.

2. **Insertion of Person Data**:
   - Three person records are created and inserted into the hash table. Each record contains an SSN, name, and address.

3. **Searching for Person Data**:
   - The code searches for specific person records using their SSNs. If found, the person's information is displayed; otherwise, a "Failed to Search" message is shown.

4. **Deletion of Person Data**:
   - The code deletes person records from the hash table using their SSNs. Once deleted, the memory allocated for the person's data is freed.

5. **Cleanup**:
   - The hash table is deallocated, and the program terminates.

---

### **Algorithms and Data Structures Used**
1. **Hash Table**:
   - A hash table is a data structure that stores key-value pairs. It uses a hash function to compute an index (or "bucket") where the value should be stored or retrieved.
   - In this code, the hash function (`MyHashFunc`) calculates the index by taking the modulo of the SSN with 100 (`k % 100`). This ensures that the index falls within the range of the hash table's size.

2. **Collision Handling**:
   - The code does not explicitly show how collisions (when two keys hash to the same index) are handled. However, in a typical hash table implementation, collisions can be resolved using techniques like **chaining** (storing multiple values in a linked list at the same index) or **open addressing** (finding another available index).

3. **Dynamic Memory Management**:
   - The code uses dynamic memory allocation (e.g., `malloc` and `free`) to create and destroy person records. This ensures that memory is used efficiently and released when no longer needed.

---

### **Overall Structure**
The code is structured as follows:
1. **Header Files**:
   - `#include "Hash_Table.h"`: Includes the custom header file for the hash table implementation.
   - `#include <stdio.h>` and `#include <stdlib.h>`: Standard libraries for input/output and memory management.

2. **Hash Function**:
   - `MyHashFunc(int k)`: A simple hash function that returns the modulo of the key (`k`) with 100.

3. **Main Function**:
   - Initializes the hash table.
   - Creates and inserts person records into the hash table.
   - Searches for specific records using their SSNs.
   - Deletes records and frees their memory.
   - Deallocates the hash table and exits.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The hash table is initialized with the `TBLinit` function, which sets up the table and assigns the hash function (`MyHashFunc`).

2. **Insertion**:
   - Person records are created using `MakePersonData` and inserted into the hash table using `TBLInsert`. The hash function determines the index where each record is stored.

3. **Searching**:
   - The `TBLSearch` function is used to find records by their SSNs. If a record is found, its information is displayed using `ShowPerInfo`.

4. **Deletion**:
   - The `TBLDelete` function removes records from the hash table using their SSNs. The memory allocated for the deleted records is freed using `free`.

5. **Cleanup**:
   - The program ends by deallocating the hash table and freeing any remaining resources.

---

### **Problem Being Solved**
The code solves the problem of **efficiently managing and retrieving person records** using a hash table. Hash tables are ideal for this purpose because they provide fast insertion, lookup, and deletion operations (average time complexity of O(1)).

---

### **Approach Taken**
1. **Custom Hash Function**:
   - The hash function (`MyHashFunc`) is simple and uses modulo arithmetic to distribute keys across the hash table.

2. **Modular Design**:
   - The code is modular, with separate functions for initialization, insertion, searching, and deletion. This makes the code easy to understand and maintain.

3. **Dynamic Memory Management**:
   - Memory is allocated and freed dynamically, ensuring efficient use of resources.

4. **Error Handling**:
   - The code checks for `NULL` pointers after searching and deleting records, preventing crashes and memory leaks.

---

### **Summary**
This code demonstrates how a hash table can be used to store and manage person records efficiently. It uses a custom hash function, dynamic memory management, and modular design to achieve its goals. The hash table provides fast access to data, making it suitable for applications where quick lookups and modifications are required.

Let me know if you'd like a line-by-line explanation or suggestions for improvements!