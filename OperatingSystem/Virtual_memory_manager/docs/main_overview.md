# Code Overview: main.cpp

This C++ code implements a **virtual memory management system**, which is a fundamental component of modern operating systems. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code simulates a **virtual memory system** that manages the translation of virtual addresses (used by programs) to physical addresses (used by the hardware). It includes:
1. **Page Table**: A data structure that maps virtual pages to physical frames.
2. **Translation Lookaside Buffer (TLB)**: A cache for recently used page table entries to speed up address translation.
3. **Memory Management**: Handles page faults, page replacement, and memory allocation.

The system solves the problem of **efficiently managing memory** in a computer system where the virtual address space (used by programs) is larger than the physical memory (RAM). It ensures that programs can use a large virtual address space while efficiently utilizing limited physical memory.

---

### **Main Functionality**
The code provides the following core functionalities:
1. **Address Translation**:
   - Converts virtual addresses (used by programs) into physical addresses (used by hardware).
   - Uses a **page table** to map virtual page numbers to physical frame numbers.
   - Implements a **TLB** to cache frequently used translations for faster access.

2. **Page Replacement**:
   - When physical memory is full, the system must evict a page to make room for a new one.
   - The code supports **LRU (Least Recently Used)** replacement policy, which evicts the page that hasn't been accessed for the longest time.

3. **Thread Safety**:
   - The system is designed to be thread-safe using **mutexes** to protect shared data structures like the page table and TLB.

4. **Memory Management**:
   - Tracks whether pages are in physical memory or stored in a **backing store** (disk).
   - Handles **dirty pages** (pages that have been modified and need to be written back to disk).

---

### **Algorithms Used**
1. **Page Table Lookup**:
   - The page table is implemented as a vector of `PageTableEntry` objects.
   - Each entry contains metadata about the page (e.g., valid bit, dirty bit, frame number).

2. **TLB Lookup**:
   - The TLB is implemented as a combination of a **list** (for LRU ordering) and a **hash map** (for O(1) lookups).
   - When a TLB miss occurs, the system falls back to the page table.

3. **LRU Replacement**:
   - Both the TLB and page table use the LRU algorithm to evict the least recently used entry when the cache is full.
   - The `lastAccessed` timestamp is used to track when a page or TLB entry was last used.

---

### **Overall Structure**
The code is organized into several key components:

1. **Core Data Types**:
   - Defines types like `VirtualAddress`, `PhysicalAddress`, `PageNumber`, and `FrameNumber` to represent memory addresses and page/frame numbers.

2. **Helper Functions**:
   - `getPageNumber`: Extracts the page number from a virtual address.
   - `getOffset`: Extracts the byte offset within a page.
   - `makePhysicalAddress`: Combines a frame number and offset into a physical address.

3. **Page Table**:
   - The `PageTable` class manages the mapping of virtual pages to physical frames.
   - Each `PageTableEntry` contains metadata about the page (e.g., valid bit, dirty bit, frame number).

4. **TLB**:
   - The `TLB` class caches recently used page table entries for faster address translation.
   - It uses a combination of a list (for LRU ordering) and a hash map (for fast lookups).

5. **Thread Safety**:
   - Both the page table and TLB use **mutexes** to ensure thread-safe access to shared data.

---

### **How the Parts Work Together**
1. **Address Translation**:
   - When a program accesses a virtual address, the system first checks the TLB.
   - If the TLB contains the translation (TLB hit), it returns the physical address.
   - If the TLB does not contain the translation (TLB miss), the system consults the page table.
   - If the page table entry is valid, the system updates the TLB with the new translation.
   - If the page table entry is invalid, a **page fault** occurs, and the system must load the page from the backing store.

2. **Page Replacement**:
   - When physical memory is full, the system uses the LRU algorithm to evict a page.
   - If the evicted page is dirty, it is written back to the backing store.
   - The new page is loaded into the freed frame, and the page table and TLB are updated.

3. **Thread Safety**:
   - Mutexes ensure that multiple threads can safely access the page table and TLB without causing race conditions.

---

### **Problem Being Solved**
The code solves the problem of **efficient memory management** in a system with limited physical memory. It allows programs to use a large virtual address space while ensuring that frequently used pages are kept in physical memory for fast access. The TLB and page table work together to minimize the overhead of address translation, and the LRU algorithm ensures that the most relevant pages are kept in memory.

---

### **Approach Taken**
The code takes a **modular and object-oriented approach**:
- Each component (page table, TLB) is encapsulated in its own class.
- Helper functions handle low-level address manipulation.
- Thread safety is ensured using mutexes.
- The system is designed to be extensible, allowing for different page replacement algorithms or TLB implementations.

---

### **Summary**
This code implements a virtual memory management system that:
1. Translates virtual addresses to physical addresses using a page table and TLB.
2. Manages memory efficiently using the LRU replacement algorithm.
3. Ensures thread safety with mutexes.
4. Simulates the behavior of a real operating system's memory management unit (MMU).

This is a foundational piece of code that could be extended to include features like multi-level page tables, different replacement algorithms, or support for larger memory sizes.