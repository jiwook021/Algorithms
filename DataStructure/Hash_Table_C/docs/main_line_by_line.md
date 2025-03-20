# Step-by-Step Explanation: main.c

Absolutely! Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, the logic behind it, and why certain techniques are used.

---

### **1. Header Files**
```c
#include "Hash_Table.h"
#include <stdio.h>
#include <stdlib.h>
```

#### **What It Does**
- These lines include external libraries and header files that the program needs to run.

#### **Explanation**
1. **`#include "Hash_Table.h"`**:
   - This includes a custom header file (`Hash_Table.h`) that likely contains the definitions for the hash table data structure and its associated functions (e.g., `TBLinit`, `TBLInsert`, `TBLSearch`, `TBLDelete`).
   - Think of this as importing a toolbox with tools (functions) and materials (data structures) that the program will use.

2. **`#include <stdio.h>`**:
   - This includes the **Standard Input/Output** library, which provides functions like `printf` for printing to the console and `scanf` for reading input.

3. **`#include <stdlib.h>`**:
   - This includes the **Standard Library**, which provides functions for memory management (e.g., `malloc` and `free`) and other utilities.

#### **Why It’s Used**
- These libraries are essential for the program to perform basic tasks like printing output, managing memory, and using the hash table.

---

### **2. Hash Function**
```c
int MyHashFunc(int k)
{
    return k % 100;
}
```

#### **What It Does**
- This is a **hash function** that takes an integer key (`k`) and returns an index (a number between 0 and 99) where the corresponding data should be stored in the hash table.

#### **Explanation**
1. **Hash Function**:
   - A hash function is a mathematical function that takes an input (in this case, an SSN) and maps it to a fixed-size output (an index in the hash table).
   - The goal is to distribute keys evenly across the hash table to minimize collisions (when two keys map to the same index).

2. **`k % 100`**:
   - The modulo operator (`%`) calculates the remainder when `k` is divided by 100.
   - For example:
     - If `k = 11111111`, then `11111111 % 100 = 11`. So, the data will be stored at index 11.
     - If `k = 22222222`, then `22222222 % 100 = 22`. So, the data will be stored at index 22.

#### **Why It’s Used**
- The hash function ensures that the data is distributed evenly across the hash table, making it faster to insert, search, and delete records.

---

### **3. Main Function**
```c
int main()
{
    Table myTbl;
    Person *np;
    Person *sp;
    Person *rp;
```

#### **What It Does**
- This is the entry point of the program. It declares variables and initializes the hash table.

#### **Explanation**
1. **`Table myTbl;`**:
   - Declares a variable `myTbl` of type `Table`. This represents the hash table where person records will be stored.

2. **`Person *np;`**, **`Person *sp;`**, **`Person *rp;`**:
   - These are pointers to `Person` structures. They will be used to:
     - `np`: Store newly created person records.
     - `sp`: Store search results.
     - `rp`: Store records to be deleted.

#### **Why It’s Used**
- These variables are necessary to manage the data and interact with the hash table.

---

### **4. Hash Table Initialization**
```c
TBLinit(&myTbl, MyHashFunc);
```

#### **What It Does**
- Initializes the hash table (`myTbl`) and assigns the hash function (`MyHashFunc`) to it.

#### **Explanation**
1. **`TBLinit`**:
   - This function (likely defined in `Hash_Table.h`) sets up the hash table for use.
   - It prepares the internal structure of the hash table (e.g., allocating memory for buckets).

2. **`&myTbl`**:
   - The `&` operator passes the address of `myTbl` to the function, allowing the function to modify the actual hash table.

3. **`MyHashFunc`**:
   - The hash function is passed to the hash table so it knows how to calculate indices for keys.

#### **Why It’s Used**
- Initialization is necessary to prepare the hash table for use. Without it, the table would not be ready to store data.

---

### **5. Inserting Data into the Hash Table**
```c
np = MakePersonData(11111111, "kim", "Seoul");
TBLInsert(&myTbl, GetSSN(np), np);
```

#### **What It Does**
- Creates a person record and inserts it into the hash table.

#### **Explanation**
1. **`MakePersonData`**:
   - This function (likely defined in `Hash_Table.h`) creates a `Person` structure with the provided SSN, name, and address.
   - For example:
     ```c
     Person *p = MakePersonData(11111111, "kim", "Seoul");
     ```
     creates a person with SSN `11111111`, name `"kim"`, and address `"Seoul"`.

2. **`TBLInsert`**:
   - This function inserts the person record into the hash table.
   - It uses the hash function (`MyHashFunc`) to calculate the index where the record should be stored.

3. **`GetSSN(np)`**:
   - This function retrieves the SSN from the `Person` structure. The SSN is used as the key for the hash table.

#### **Why It’s Used**
- Insertion is necessary to add data to the hash table so it can be searched or deleted later.

---

### **6. Searching for Data**
```c
sp = TBLSearch(&myTbl, searchname1);
if (sp != NULL)
{
    printf("\nSearched: ");
    ShowPerInfo(sp);
}
else
{
    printf("Failed to Search %d ", searchname1);
}
```

#### **What It Does**
- Searches for a person record in the hash table using their SSN.

#### **Explanation**
1. **`TBLSearch`**:
   - This function searches the hash table for a record with the specified SSN (`searchname1`).
   - It uses the hash function to calculate the index where the record should be located.

2. **`if (sp != NULL)`**:
   - If the search is successful (`sp` points to a valid record), the program prints the person’s information using `ShowPerInfo`.

3. **`else`**:
   - If the search fails (`sp` is `NULL`), the program prints a "Failed to Search" message.

#### **Why It’s Used**
- Searching allows the program to retrieve specific records from the hash table.

---

### **7. Deleting Data**
```c
rp = TBLDelete(&myTbl, 11111111);
printf("deleted info of %s\n", rp->name);
if (rp != NULL)
    free(rp);
```

#### **What It Does**
- Deletes a person record from the hash table and frees the memory allocated for it.

#### **Explanation**
1. **`TBLDelete`**:
   - This function removes the record with the specified SSN from the hash table.

2. **`free(rp)`**:
   - The `free` function deallocates the memory previously allocated for the `Person` structure.

#### **Why It’s Used**
- Deletion is necessary to remove records that are no longer needed and free up memory.

---

### **8. Cleanup**
```c
printf("\nDeallocated Hash Table\n");
return 0;
```

#### **What It Does**
- Prints a message indicating that the hash table has been deallocated and ends the program.

#### **Why It’s Used**
- Cleanup ensures that all resources are properly released before the program exits.

---

### **Summary**
This code demonstrates how to:
1. Initialize a hash table.
2. Insert, search, and delete records.
3. Use dynamic memory management to allocate and free memory.

Each step is carefully designed to ensure efficient data management and retrieval. Let me know if you’d like further clarification or a deeper dive into any specific part!