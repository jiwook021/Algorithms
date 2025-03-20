# Code Overview: Hash_Table.c

This C code implements a **hash table**, which is a fundamental data structure used for efficient data storage and retrieval. Let's break down the purpose, functionality, and structure of this code in detail.

---

### **Purpose of the Code**
The code implements a hash table to store and manage **Person** records. Each record contains:
- A **Social Security Number (SSN)** as the key.
- A **name** and **address** as the value.

The hash table allows for:
1. **Fast insertion** of records.
2. **Efficient searching** for records using the SSN as the key.
3. **Deletion** of records when they are no longer needed.

The hash table is designed to handle collisions implicitly by using a fixed-size array (with `MAX_TBL` slots) and a **hash function** to map keys (SSNs) to specific slots in the array.

---

### **Main Functionality**
The code provides the following key functionalities:
1. **Initialization**: Prepares the hash table for use by setting all slots to an "EMPTY" state and assigning a hash function.
2. **Insertion**: Adds a new `Person` record to the hash table.
3. **Deletion**: Marks a record as "DELETED" but does not physically remove it from memory.
4. **Searching**: Retrieves a `Person` record based on the SSN key.
5. **Utility Functions**: Helper functions to create `Person` records and display their information.

---

### **Algorithms Used**
1. **Hash Function**:
   - The hash function (`hf`) is provided externally (passed as a function pointer) and is responsible for mapping a key (SSN) to an index in the hash table.
   - The hash function ensures that keys are distributed uniformly across the table, minimizing collisions.

2. **Open Addressing**:
   - The hash table uses **open addressing** to handle collisions. If a slot is already occupied (status = INUSE), the code does not attempt to resolve the collision (e.g., by chaining or probing). This is a limitation of the current implementation.

3. **Lazy Deletion**:
   - When a record is deleted, its status is marked as "DELETED" instead of physically removing it. This allows the slot to be reused later.

---

### **Overall Structure**
The code is organized into several components:
1. **Data Structures**:
   - `Table`: Represents the hash table. It contains:
     - An array of `Slot` structures (`tbl`), where each `Slot` holds a key, value, and status.
     - A function pointer (`hf`) to the hash function.
   - `Person`: Represents a record with an SSN, name, and address.

2. **Functions**:
   - `TBLinit`: Initializes the hash table.
   - `MakePersonData`: Creates a new `Person` record.
   - `TBLInsert`: Inserts a `Person` record into the hash table.
   - `TBLDelete`: Marks a record as deleted.
   - `TBLSearch`: Searches for a record by key.
   - `ShowPerInfo`: Displays the details of a `Person` record.

---

### **How the Code Works Together**
1. **Initialization**:
   - The `TBLinit` function sets up the hash table by marking all slots as "EMPTY" and assigning the hash function.

2. **Insertion**:
   - When a new `Person` record is created using `MakePersonData`, it is inserted into the hash table using `TBLInsert`.
   - The hash function computes the index for the record, and the record is stored in the corresponding slot.

3. **Searching**:
   - To find a record, `TBLSearch` uses the hash function to locate the slot where the record should be stored. If the slot is marked as "INUSE," the record is returned.

4. **Deletion**:
   - `TBLDelete` marks a slot as "DELETED" but does not free the memory. The record can still be accessed if needed.

5. **Display**:
   - The `ShowPerInfo` function is used to print the details of a `Person` record.

---

### **Problem Being Solved**
The code solves the problem of **efficiently managing a collection of records** (in this case, `Person` records) by using a hash table. Hash tables are ideal for scenarios where:
- Fast lookups, insertions, and deletions are required.
- The data can be uniquely identified by a key (e.g., SSN).

---

### **Approach Taken**
The approach taken in this code is straightforward:
1. Use a fixed-size array to represent the hash table.
2. Use a hash function to map keys to array indices.
3. Handle collisions implicitly by overwriting existing records (this is a limitation).
4. Use lazy deletion to mark records as deleted without freeing memory.

---

### **Limitations**
1. **No Collision Resolution**:
   - The code does not handle collisions explicitly. If two keys hash to the same index, the second record will overwrite the first.
2. **Fixed Table Size**:
   - The hash table has a fixed size (`MAX_TBL`), which can lead to inefficiencies if the table becomes too full.
3. **Memory Leaks**:
   - The code does not free memory for deleted records, which could lead to memory leaks over time.
4. **No Error Handling**:
   - The code does not handle edge cases, such as inserting into a full table or searching for a non-existent key.

---

### **Summary**
This code implements a basic hash table for managing `Person` records. It provides core functionalities like insertion, deletion, and searching but lacks advanced features like collision resolution and dynamic resizing. The code is a good starting point for understanding hash tables but would need enhancements for real-world use.

Let me know if you'd like to dive deeper into any specific part of the code!