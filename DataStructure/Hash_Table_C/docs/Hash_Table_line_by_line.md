# Step-by-Step Explanation: Hash_Table.c

Absolutely! Let’s break down the code **line by line** and **section by section**, explaining everything in detail. I’ll use simple language, analogies, and examples to make it easy to understand, even for beginners.

---

### **1. Header Files and Includes**
```c
#include "Hash_Table.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
```

#### **What It Does**
- These lines include necessary libraries and a custom header file (`Hash_Table.h`).
- `stdio.h` is for input/output functions like `printf`.
- `string.h` is for string manipulation functions like `strcpy`.
- `stdlib.h` is for memory management functions like `malloc`.

#### **Why It’s Used**
- These libraries provide tools for working with strings, memory, and printing to the console, which are essential for this program.

---

### **2. Table Initialization Function: `TBLinit`**
```c
void TBLinit(Table *pt, HashFunc* f)
{
    for (int i = 0; i < MAX_TBL; i++)
        (pt->tbl[i]).status = EMPTY;

    pt->hf = f;
}
```

#### **What It Does**
- This function initializes a hash table.
- It sets the status of every slot in the table to `EMPTY`.
- It assigns a hash function (`f`) to the table.

#### **Breakdown**
1. **`Table *pt`**:
   - `pt` is a pointer to a `Table` structure. Think of it as a "handle" to the hash table.
   - The `Table` structure contains:
     - An array of `Slot` structures (`tbl`), where each `Slot` holds a key, value, and status.
     - A function pointer (`hf`) to the hash function.

2. **`for (int i = 0; i < MAX_TBL; i++)`**:
   - This loop iterates over every slot in the table.
   - `MAX_TBL` is the size of the table (number of slots).

3. **`(pt->tbl[i]).status = EMPTY;`**:
   - For each slot, the `status` is set to `EMPTY`.
   - This means the slot is available for storing data.

4. **`pt->hf = f;`**:
   - The hash function (`f`) is assigned to the table.
   - The hash function will later be used to compute the index for storing and retrieving data.

#### **Why It’s Used**
- Initialization ensures the table starts in a clean state, with all slots empty and ready for use.
- Assigning the hash function allows the table to compute indices for keys.

#### **Example**
Imagine a bookshelf with 10 shelves (`MAX_TBL = 10`). Before you start placing books, you make sure all shelves are empty. This is what `TBLinit` does.

---

### **3. Creating a Person Record: `MakePersonData`**
```c
Person * MakePersonData(int ssn, char *name, char *addr)
{
    Person *newP = (Person*) malloc(sizeof(Person));
    newP->ssn = ssn; 
    strcpy(newP->name, name);
    strcpy(newP->addr, addr);
    return newP;
}
```

#### **What It Does**
- This function creates a new `Person` record.
- It allocates memory for the record, assigns the SSN, name, and address, and returns a pointer to the record.

#### **Breakdown**
1. **`Person *newP = (Person*) malloc(sizeof(Person));`**:
   - `malloc` allocates memory for a `Person` structure.
   - `sizeof(Person)` calculates the size of the `Person` structure in bytes.
   - `(Person*)` casts the memory address to a `Person` pointer.

2. **`newP->ssn = ssn;`**:
   - Assigns the SSN to the `ssn` field of the `Person` structure.

3. **`strcpy(newP->name, name);`**:
   - Copies the `name` string into the `name` field of the `Person` structure.
   - `strcpy` is used because strings in C are arrays of characters and cannot be assigned directly.

4. **`strcpy(newP->addr, addr);`**:
   - Copies the `addr` string into the `addr` field of the `Person` structure.

5. **`return newP;`**:
   - Returns the pointer to the newly created `Person` record.

#### **Why It’s Used**
- This function encapsulates the creation of a `Person` record, making the code modular and reusable.

#### **Example**
Think of this function as a factory that produces "Person" objects. You provide the details (SSN, name, address), and it gives you a fully constructed object.

---

### **4. Inserting a Record: `TBLInsert`**
```c
void TBLInsert(Table *pt, Key k, Value v)
{
    int hv = pt->hf(k);
    pt->tbl[hv].key = k;
    pt->tbl[hv].val = v;
    pt->tbl[hv].status = INUSE;

    printf("Inserted SSN: %d ", v->ssn);
    printf("Name: %s ", v->name);
    printf("Address: %s \n", v->addr);
}
```

#### **What It Does**
- This function inserts a `Person` record into the hash table.
- It uses the hash function to compute the index for the record and stores the record at that index.

#### **Breakdown**
1. **`int hv = pt->hf(k);`**:
   - The hash function (`hf`) is called with the key (`k`) to compute the index (`hv`).

2. **`pt->tbl[hv].key = k;`**:
   - The key is stored in the slot at index `hv`.

3. **`pt->tbl[hv].val = v;`**:
   - The value (`v`, which is a `Person` record) is stored in the slot.

4. **`pt->tbl[hv].status = INUSE;`**:
   - The status of the slot is set to `INUSE`, indicating it is occupied.

5. **`printf` Statements**:
   - Print the details of the inserted record.

#### **Why It’s Used**
- This function adds records to the hash table, enabling fast lookups later.

#### **Example**
Imagine you’re placing a book on a shelf. The hash function tells you which shelf to use, and you place the book there.

---

### **5. Searching for a Record: `TBLSearch`**
```c
Value TBLSearch(Table *pt, Key k)
{
    int hv = pt->hf(k);

    if ((pt->tbl[hv]).status != INUSE)
        return NULL;
    else
        return (pt->tbl[hv]).val;
}
```

#### **What It Does**
- This function searches for a record in the hash table using the key.
- It returns the record if found, or `NULL` if not.

#### **Breakdown**
1. **`int hv = pt->hf(k);`**:
   - The hash function computes the index for the key.

2. **`if ((pt->tbl[hv]).status != INUSE)`**:
   - Checks if the slot is not in use (either `EMPTY` or `DELETED`).
   - If so, returns `NULL`.

3. **`return (pt->tbl[hv]).val;`**:
   - If the slot is in use, returns the value stored there.

#### **Why It’s Used**
- This function enables fast retrieval of records using the key.

#### **Example**
Imagine looking for a book on a shelf. The hash function tells you which shelf to check. If the book is there, you take it; otherwise, you conclude it’s not available.

---

### **6. Deleting a Record: `TBLDelete`**
```c
Value TBLDelete(Table *pt, Key k)
{
    int hv = pt->hf(k);

    if ((pt->tbl[hv]).status != INUSE)
        return NULL;
    else
    {
        (pt->tbl[hv]).status = DELETED;
        return (pt->tbl[hv]).val;
    }
}
```

#### **What It Does**
- This function marks a record as deleted but does not free the memory.
- It returns the deleted record.

#### **Breakdown**
1. **`int hv = pt->hf(k);`**:
   - The hash function computes the index for the key.

2. **`if ((pt->tbl[hv]).status != INUSE)`**:
   - Checks if the slot is not in use.
   - If so, returns `NULL`.

3. **`(pt->tbl[hv]).status = DELETED;`**:
   - Marks the slot as `DELETED`.

4. **`return (pt->tbl[hv]).val;`**:
   - Returns the value stored in the slot.

#### **Why It’s Used**
- This function allows for lazy deletion, which is faster than physically removing the record.

#### **Example**
Imagine marking a book as "unavailable" on a shelf without actually removing it. This way, you can still access it if needed.

---

### **7. Displaying a Record: `ShowPerInfo`**
```c
void ShowPerInfo(Person *p)
{
    printf("SSN: %d \n", p->ssn);
    printf("Name: %s \n", p->name);
    printf("Address: %s \n\n", p->addr);
}
```

#### **What It Does**
- This function prints the details of a `Person` record.

#### **Breakdown**
1. **`printf("SSN: %d \n", p->ssn);`**:
   - Prints the SSN.

2. **`printf("Name: %s \n", p->name);`**:
   - Prints the name.

3. **`printf("Address: %s \n\n", p->addr);`**:
   - Prints the address.

#### **Why It’s Used**
- This function provides a way to view the details of a record.

#### **Example**
Think of this as reading the details of a book (title, author, etc.) from its cover.

---

### **Summary**
This code implements a basic hash table for managing `Person` records. It provides core functionalities like initialization, insertion, deletion, and searching. While it’s a good starting point, it has limitations like no collision resolution and fixed table size. Let me know if you’d like to dive deeper into any specific part!