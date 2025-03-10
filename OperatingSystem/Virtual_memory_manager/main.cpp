#include <iostream>
#include <vector>
#include <queue>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <random>

// Core data types
using VirtualAddress = uint32_t;  // 32-bit virtual address
using PhysicalAddress = uint32_t; // 32-bit physical address
using PageNumber = uint32_t;      // Virtual page number
using FrameNumber = uint32_t;     // Physical frame number
using Byte = uint8_t;             // Basic storage unit

// Configuration constants
constexpr size_t PAGE_SIZE = 4096;                // 4KB pages
constexpr size_t VIRTUAL_MEMORY_SIZE = 1UL << 32; // 4GB virtual memory
constexpr size_t PHYSICAL_MEMORY_SIZE = 1UL << 24; // 16MB physical memory
constexpr size_t TLB_SIZE = 16;                   // 16 entries in TLB
constexpr size_t BACKING_STORE_SIZE = 1UL << 28;  // 256MB backing store

// Helper functions for address translation
PageNumber getPageNumber(VirtualAddress addr) {
    return addr / PAGE_SIZE;
}

size_t getOffset(VirtualAddress addr) {
    return addr % PAGE_SIZE;
}

PhysicalAddress makePhysicalAddress(FrameNumber frame, size_t offset) {
    return frame * PAGE_SIZE + offset;
}

//=============================================================================
// Page Table Implementation
//=============================================================================

class PageTableEntry {
public:
    bool valid = false;          // Is the entry valid (in physical memory)?
    bool dirty = false;          // Has the page been modified?
    bool referenced = false;     // Has the page been accessed recently?
    FrameNumber frameNumber = 0; // Physical frame number
    
    // For tracking pages in backing store
    bool inBackingStore = false;
    size_t backingStoreLocation = 0;
    
    // For page replacement algorithms
    std::chrono::steady_clock::time_point lastAccessed; // For LRU
};

class PageTable {
private:
    std::vector<PageTableEntry> entries;
    mutable std::mutex tableMutex;  // For thread safety

public:
    // Constructor: Initialize with the number of virtual pages
    PageTable(size_t numPages) : entries(numPages) {}
    
    // Get entry for a virtual page (thread-safe)
    PageTableEntry getEntry(PageNumber pageNum) const {
        std::lock_guard<std::mutex> lock(tableMutex);
        if (pageNum >= entries.size()) {
            throw std::out_of_range("Page number out of range");
        }
        return entries[pageNum];
    }
    
    // Update entry for a virtual page (thread-safe)
    void updateEntry(PageNumber pageNum, const PageTableEntry& entry) {
        std::lock_guard<std::mutex> lock(tableMutex);
        if (pageNum >= entries.size()) {
            throw std::out_of_range("Page number out of range");
        }
        entries[pageNum] = entry;
    }
    
    // Mark page as accessed (for LRU algorithm)
    void markAccessed(PageNumber pageNum) {
        std::lock_guard<std::mutex> lock(tableMutex);
        if (pageNum >= entries.size()) {
            throw std::out_of_range("Page number out of range");
        }
        entries[pageNum].referenced = true;
        entries[pageNum].lastAccessed = std::chrono::steady_clock::now();
    }
    
    // Mark page as dirty
    void markDirty(PageNumber pageNum) {
        std::lock_guard<std::mutex> lock(tableMutex);
        if (pageNum >= entries.size()) {
            throw std::out_of_range("Page number out of range");
        }
        entries[pageNum].dirty = true;
    }
    
    // Size accessor
    size_t size() const {
        return entries.size();
    }
};

//=============================================================================
// Translation Lookaside Buffer (TLB)
//=============================================================================

// TLB entry structure
class TLBEntry {
public:
    PageNumber pageNumber;
    FrameNumber frameNumber;
    std::chrono::steady_clock::time_point lastAccessed;
    
    TLBEntry(PageNumber page, FrameNumber frame) 
        : pageNumber(page), frameNumber(frame), 
          lastAccessed(std::chrono::steady_clock::now()) {}
};

class TLB {
private:
    // List for LRU ordering (front is most recent)
    mutable std::list<TLBEntry> entries;  
    
    // Map for O(1) lookups
    std::unordered_map<PageNumber, std::list<TLBEntry>::iterator> pageMap;
    
    size_t capacity;
    mutable std::mutex tlbMutex;  // For thread safety
    
public:
    TLB(size_t size) : capacity(size) {}
    
    // Look up a virtual page in the TLB (thread-safe)
    bool lookup(PageNumber pageNum, FrameNumber& frameNum) {
        std::lock_guard<std::mutex> lock(tlbMutex);
        auto it = pageMap.find(pageNum);
        if (it != pageMap.end()) {
            // TLB hit
            frameNum = it->second->frameNumber;
            
            // Update access time (LRU policy)
            it->second->lastAccessed = std::chrono::steady_clock::now();
            
            // Move to front of list (most recently used)
            entries.splice(entries.begin(), entries, it->second);
            return true;
        }
        // TLB miss
        return false;
    }
    
    // Add or update an entry in the TLB (thread-safe)
    void update(PageNumber pageNum, FrameNumber frameNum) {
        std::lock_guard<std::mutex> lock(tlbMutex);
        
        // Check if entry already exists
        auto it = pageMap.find(pageNum);
        if (it != pageMap.end()) {
            // Update existing entry
            it->second->frameNumber = frameNum;
            it->second->lastAccessed = std::chrono::steady_clock::now();
            
            // Move to front of list (most recently used)
            entries.splice(entries.begin(), entries, it->second);
        } else {
            // Need to add new entry
            if (entries.size() >= capacity) {
                // TLB is full, remove least recently used entry
                PageNumber oldPage = entries.back().pageNumber;
                pageMap.erase(oldPage);
                entries.pop_back();
            }
            
            // Add new entry
            entries.emplace_front(pageNum, frameNum);
            pageMap[pageNum] = entries.begin();
        }
    }
    
    // Invalidate an entry (e.g., when a page is swapped out)
    void invalidate(PageNumber pageNum) {
        std::lock_guard<std::mutex> lock(tlbMutex);
        auto it = pageMap.find(pageNum);
        if (it != pageMap.end()) {
            entries.erase(it->second);
            pageMap.erase(it);
        }
    }
    
    // Clear the entire TLB
    void clear() {
        std::lock_guard<std::mutex> lock(tlbMutex);
        entries.clear();
        pageMap.clear();
    }
};

//=============================================================================
// Physical Memory Simulation
//=============================================================================

class PhysicalMemory {
private:
    std::vector<Byte> memory;
    std::vector<bool> frameAllocated;
    size_t numFrames;
    mutable std::mutex memoryMutex;  // For thread safety
    
public:
    PhysicalMemory(size_t size) 
        : memory(size, 0), 
          frameAllocated(size / PAGE_SIZE, false),
          numFrames(size / PAGE_SIZE) {}
    
    // Allocate a physical frame (thread-safe)
    bool allocateFrame(FrameNumber& frameNum) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        // Find first free frame
        for (size_t i = 0; i < numFrames; ++i) {
            if (!frameAllocated[i]) {
                frameAllocated[i] = true;
                frameNum = static_cast<FrameNumber>(i);
                return true;
            }
        }
        
        // No free frames
        return false;
    }
    
    // Free a physical frame (thread-safe)
    void freeFrame(FrameNumber frameNum) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (frameNum >= numFrames) {
            throw std::out_of_range("Frame number out of range");
        }
        frameAllocated[frameNum] = false;
    }
    
    // Read a byte from physical memory (thread-safe)
    Byte readByte(PhysicalAddress addr) const {
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (addr >= memory.size()) {
            throw std::out_of_range("Physical address out of range");
        }
        return memory[addr];
    }
    
    // Write a byte to physical memory (thread-safe)
    void writeByte(PhysicalAddress addr, Byte value) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (addr >= memory.size()) {
            throw std::out_of_range("Physical address out of range");
        }
        memory[addr] = value;
    }
    
    // Read a page from physical memory into a buffer (thread-safe)
    void readPage(FrameNumber frameNum, std::vector<Byte>& buffer) const {
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (frameNum >= numFrames) {
            throw std::out_of_range("Frame number out of range");
        }
        
        size_t frameOffset = frameNum * PAGE_SIZE;
        buffer.resize(PAGE_SIZE);
        std::copy(memory.begin() + frameOffset, 
                  memory.begin() + frameOffset + PAGE_SIZE, 
                  buffer.begin());
    }
    
    // Write a page to physical memory from a buffer (thread-safe)
    void writePage(FrameNumber frameNum, const std::vector<Byte>& buffer) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (frameNum >= numFrames) {
            throw std::out_of_range("Frame number out of range");
        }
        if (buffer.size() != PAGE_SIZE) {
            throw std::invalid_argument("Buffer size must match page size");
        }
        
        size_t frameOffset = frameNum * PAGE_SIZE;
        std::copy(buffer.begin(), buffer.end(), memory.begin() + frameOffset);
    }
    
    // Get the total number of frames
    size_t getTotalFrames() const {
        return numFrames;
    }
    
    // Get the number of free frames
    size_t getFreeFrameCount() const {
        std::lock_guard<std::mutex> lock(memoryMutex);
        return std::count(frameAllocated.begin(), frameAllocated.end(), false);
    }
};

//=============================================================================
// Backing Store (Simulated Disk)
//=============================================================================

class BackingStore {
private:
    std::vector<Byte> storage;
    std::vector<bool> blockAllocated;
    size_t numBlocks;
    mutable std::mutex storeMutex;  // For thread safety
    
public:
    BackingStore(size_t size) 
        : storage(size, 0), 
          blockAllocated(size / PAGE_SIZE, false),
          numBlocks(size / PAGE_SIZE) {}
    
    // Allocate a block in the backing store (thread-safe)
    bool allocateBlock(size_t& blockNum) {
        std::lock_guard<std::mutex> lock(storeMutex);
        
        // Find first free block
        for (size_t i = 0; i < numBlocks; ++i) {
            if (!blockAllocated[i]) {
                blockAllocated[i] = true;
                blockNum = i;
                return true;
            }
        }
        
        // No free blocks
        return false;
    }
    
    // Free a block in the backing store (thread-safe)
    void freeBlock(size_t blockNum) {
        std::lock_guard<std::mutex> lock(storeMutex);
        if (blockNum >= numBlocks) {
            throw std::out_of_range("Block number out of range");
        }
        blockAllocated[blockNum] = false;
    }
    
    // Read a page from the backing store (thread-safe)
    std::vector<Byte> readPage(size_t blockNum) const {
        std::lock_guard<std::mutex> lock(storeMutex);
        if (blockNum >= numBlocks) {
            throw std::out_of_range("Block number out of range");
        }
        
        std::vector<Byte> buffer(PAGE_SIZE);
        size_t blockOffset = blockNum * PAGE_SIZE;
        std::copy(storage.begin() + blockOffset, 
                  storage.begin() + blockOffset + PAGE_SIZE, 
                  buffer.begin());
        return buffer;
    }
    
    // Write a page to the backing store (thread-safe)
    void writePage(size_t blockNum, const std::vector<Byte>& buffer) {
        std::lock_guard<std::mutex> lock(storeMutex);
        if (blockNum >= numBlocks) {
            throw std::out_of_range("Block number out of range");
        }
        if (buffer.size() != PAGE_SIZE) {
            throw std::invalid_argument("Buffer size must match page size");
        }
        
        size_t blockOffset = blockNum * PAGE_SIZE;
        std::copy(buffer.begin(), buffer.end(), storage.begin() + blockOffset);
    }
    
    // Get the total number of blocks
    size_t getTotalBlocks() const {
        return numBlocks;
    }
    
    // Get the number of free blocks
    size_t getFreeBlockCount() const {
        std::lock_guard<std::mutex> lock(storeMutex);
        return std::count(blockAllocated.begin(), blockAllocated.end(), false);
    }
};

//=============================================================================
// Page Replacement Algorithms
//=============================================================================

// Abstract interface for page replacement algorithms
class PageReplacer {
public:
    virtual ~PageReplacer() = default;
    
    // Select a page to replace and return its virtual page number
    virtual PageNumber selectVictim() = 0;
    
    // Notify the replacer that a page was accessed
    virtual void notifyAccessed(PageNumber pageNum) = 0;
    
    // Notify the replacer that a page was loaded
    virtual void notifyLoaded(PageNumber pageNum) = 0;
    
    // Notify the replacer that a page was unloaded
    virtual void notifyUnloaded(PageNumber pageNum) = 0;
};

// FIFO (First-In-First-Out) implementation
class FIFOReplacer : public PageReplacer {
private:
    std::queue<PageNumber> pageQueue;
    std::unordered_set<PageNumber> pageSet;  // For O(1) membership test
    mutable std::mutex replacerMutex;  // For thread safety
    
public:
    PageNumber selectVictim() override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        if (pageQueue.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        
        PageNumber victim = pageQueue.front();
        pageQueue.pop();
        pageSet.erase(victim);
        return victim;
    }
    
    void notifyAccessed(PageNumber pageNum) override {
        // FIFO doesn't care about accesses - no action needed
    }
    
    void notifyLoaded(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        if (pageSet.find(pageNum) == pageSet.end()) {
            pageQueue.push(pageNum);
            pageSet.insert(pageNum);
        }
    }
    
    void notifyUnloaded(PageNumber pageNum) override {
        // In a real implementation, we'd handle this more efficiently
        // For now, we'll just leave it in the queue until it reaches the front
        // (the pageSet ensures we won't add it again while it's already queued)
        std::lock_guard<std::mutex> lock(replacerMutex);
        pageSet.erase(pageNum);
    }
};

// LRU (Least Recently Used) implementation
class LRUReplacer : public PageReplacer {
private:
    // Map from page number to its position in the LRU list
    std::unordered_map<PageNumber, std::list<PageNumber>::iterator> pageMap;
    
    // LRU list, front is most recently used, back is least recently used
    mutable std::list<PageNumber> lruList;
    
    mutable std::mutex replacerMutex;  // For thread safety
    
public:
    PageNumber selectVictim() override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        if (lruList.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        
        // Least recently used page is at the back
        PageNumber victim = lruList.back();
        lruList.pop_back();
        pageMap.erase(victim);
        return victim;
    }
    
    void notifyAccessed(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        auto it = pageMap.find(pageNum);
        if (it != pageMap.end()) {
            // Move to front of LRU list (most recently used)
            lruList.erase(it->second);
            lruList.push_front(pageNum);
            it->second = lruList.begin();
        }
    }
    
    void notifyLoaded(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        auto it = pageMap.find(pageNum);
        if (it == pageMap.end()) {
            // Add to front of LRU list (most recently used)
            lruList.push_front(pageNum);
            pageMap[pageNum] = lruList.begin();
        }
    }
    
    void notifyUnloaded(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        auto it = pageMap.find(pageNum);
        if (it != pageMap.end()) {
            lruList.erase(it->second);
            pageMap.erase(it);
        }
    }
};

// Clock Algorithm implementation
class ClockReplacer : public PageReplacer {
private:
    // Map from page number to its position in the clock buffer
    std::unordered_map<PageNumber, size_t> pageToPos;
    
    // Circular buffer for clock algorithm
    std::vector<PageNumber> clockBuffer;
    
    // Reference bits for each page in the clock buffer
    std::vector<bool> referenceBit;
    
    // Current clock hand position
    size_t clockHand;
    
    mutable std::mutex replacerMutex;  // For thread safety
    
public:
    ClockReplacer(size_t maxPages) 
        : clockBuffer(maxPages, 0),
          referenceBit(maxPages, false),
          clockHand(0) {}
    
    PageNumber selectVictim() override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        if (pageToPos.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        
        // Clock algorithm: find first unreferenced page
        while (true) {
            if (!referenceBit[clockHand]) {
                // Found unreferenced page
                PageNumber victim = clockBuffer[clockHand];
                
                // Clear mapping for the victim
                pageToPos.erase(victim);
                
                // Move clock hand
                clockHand = (clockHand + 1) % clockBuffer.size();
                
                return victim;
            } else {
                // Clear reference bit and move on
                referenceBit[clockHand] = false;
                clockHand = (clockHand + 1) % clockBuffer.size();
                
                // Safety check to prevent infinite loop if all pages are referenced
                if (pageToPos.size() < clockBuffer.size() && 
                    std::all_of(referenceBit.begin(), referenceBit.end(), 
                                [](bool b) { return b; })) {
                    // All pages are referenced, just pick the current one
                    PageNumber victim = clockBuffer[clockHand];
                    pageToPos.erase(victim);
                    clockHand = (clockHand + 1) % clockBuffer.size();
                    return victim;
                }
            }
        }
    }
    
    void notifyAccessed(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        auto it = pageToPos.find(pageNum);
        if (it != pageToPos.end()) {
            // Set reference bit
            referenceBit[it->second] = true;
        }
    }
    
    void notifyLoaded(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        if (pageToPos.find(pageNum) == pageToPos.end()) {
            // Find a spot in the clock buffer
            size_t pos;
            if (pageToPos.size() < clockBuffer.size()) {
                // Still have empty spots
                pos = pageToPos.size();
            } else {
                // Need to replace an existing entry
                PageNumber victim = selectVictim();
                pos = pageToPos[victim];
                pageToPos.erase(victim);
            }
            
            // Add the new page
            clockBuffer[pos] = pageNum;
            referenceBit[pos] = true;  // Referenced when loaded
            pageToPos[pageNum] = pos;
        }
    }
    
    void notifyUnloaded(PageNumber pageNum) override {
        std::lock_guard<std::mutex> lock(replacerMutex);
        auto it = pageToPos.find(pageNum);
        if (it != pageToPos.end()) {
            // Clear reference bit
            referenceBit[it->second] = false;
            
            // Don't remove from buffer yet, selectVictim will handle it
        }
    }
};

//=============================================================================
// Virtual Memory Manager
//=============================================================================

class VirtualMemoryManager {
private:
    PageTable pageTable;
    TLB tlb;
    PhysicalMemory physicalMemory;
    BackingStore backingStore;
    std::unique_ptr<PageReplacer> pageReplacer;
    
    mutable std::mutex managerMutex;  // For thread safety
    
    // Statistics
    mutable size_t pageHits = 0;
    mutable size_t pageFaults = 0;
    mutable size_t tlbHits = 0;
    mutable size_t tlbMisses = 0;
    
public:
    VirtualMemoryManager(
        size_t numVirtualPages,
        size_t physicalMemorySize,
        size_t backingStoreSize,
        size_t tlbSize,
        std::unique_ptr<PageReplacer> replacer)
        : pageTable(numVirtualPages),
          tlb(tlbSize),
          physicalMemory(physicalMemorySize),
          backingStore(backingStoreSize),
          pageReplacer(std::move(replacer)) {}
    
    // Read a byte from virtual memory - Removed const as it modifies state
    Byte readByte(VirtualAddress addr) {
        PageNumber pageNum = getPageNumber(addr);
        size_t offset = getOffset(addr);
        
        // Try to translate address using TLB
        FrameNumber frameNum;
        bool tlbHit = tlb.lookup(pageNum, frameNum);
        
        if (tlbHit) {
            // TLB hit
            tlbHits++;
            pageReplacer->notifyAccessed(pageNum);
            PhysicalAddress physAddr = makePhysicalAddress(frameNum, offset);
            return physicalMemory.readByte(physAddr);
        }
        
        // TLB miss, check page table
        tlbMisses++;
        PageTableEntry entry = pageTable.getEntry(pageNum);
        
        if (entry.valid) {
            // Page is in memory
            pageHits++;
            frameNum = entry.frameNumber;
            pageReplacer->notifyAccessed(pageNum);
            
            // Update TLB
            tlb.update(pageNum, frameNum);
            
            PhysicalAddress physAddr = makePhysicalAddress(frameNum, offset);
            return physicalMemory.readByte(physAddr);
        } else {
            // Page fault
            pageFaults++;
            handlePageFault(pageNum);
            
            // Try again after handling fault
            return readByte(addr);
        }
    }
    
    // Write a byte to virtual memory
    void writeByte(VirtualAddress addr, Byte value) {
        PageNumber pageNum = getPageNumber(addr);
        size_t offset = getOffset(addr);
        
        // Try to translate address using TLB
        FrameNumber frameNum;
        bool tlbHit = tlb.lookup(pageNum, frameNum);
        
        if (tlbHit) {
            // TLB hit
            tlbHits++;
            pageReplacer->notifyAccessed(pageNum);
            
            // Mark page as dirty
            PageTableEntry entry = pageTable.getEntry(pageNum);
            entry.dirty = true;
            pageTable.updateEntry(pageNum, entry);
            
            PhysicalAddress physAddr = makePhysicalAddress(frameNum, offset);
            physicalMemory.writeByte(physAddr, value);
            return;
        }
        
        // TLB miss, check page table
        tlbMisses++;
        PageTableEntry entry = pageTable.getEntry(pageNum);
        
        if (entry.valid) {
            // Page is in memory
            pageHits++;
            frameNum = entry.frameNumber;
            pageReplacer->notifyAccessed(pageNum);
            
            // Mark page as dirty
            entry.dirty = true;
            pageTable.updateEntry(pageNum, entry);
            
            // Update TLB
            tlb.update(pageNum, frameNum);
            
            PhysicalAddress physAddr = makePhysicalAddress(frameNum, offset);
            physicalMemory.writeByte(physAddr, value);
        } else {
            // Page fault
            pageFaults++;
            handlePageFault(pageNum);
            
            // Try again after handling fault
            writeByte(addr, value);
        }
    }
    
    // Handle a page fault - Removed const as it modifies state
    void handlePageFault(PageNumber pageNum) {
        std::lock_guard<std::mutex> lock(managerMutex);
        
        // Check page table entry
        PageTableEntry entry = pageTable.getEntry(pageNum);
        
        // Allocate a physical frame
        FrameNumber frameNum;
        bool frameAllocated = physicalMemory.allocateFrame(frameNum);
        
        if (!frameAllocated) {
            // No free frames, need to replace a page
            PageNumber victimPageNum = pageReplacer->selectVictim();
            PageTableEntry victimEntry = pageTable.getEntry(victimPageNum);
            frameNum = victimEntry.frameNumber;
            
            // If dirty, write to backing store
            if (victimEntry.dirty) {
                std::vector<Byte> pageData;
                physicalMemory.readPage(frameNum, pageData);
                
                size_t backingStoreLocation;
                if (victimEntry.inBackingStore) {
                    // Update existing backing store location
                    backingStoreLocation = victimEntry.backingStoreLocation;
                } else {
                    // Allocate new backing store location
                    if (!backingStore.allocateBlock(backingStoreLocation)) {
                        throw std::runtime_error("Out of backing store space");
                    }
                }
                
                backingStore.writePage(backingStoreLocation, pageData);
                
                // Update victim's page table entry
                victimEntry.valid = false;
                victimEntry.inBackingStore = true;
                victimEntry.backingStoreLocation = backingStoreLocation;
                pageTable.updateEntry(victimPageNum, victimEntry);
                
                // Invalidate TLB entry for victim
                tlb.invalidate(victimPageNum);
                
                // Notify page replacer
                pageReplacer->notifyUnloaded(victimPageNum);
            } else {
                // Not dirty, just invalidate
                victimEntry.valid = false;
                pageTable.updateEntry(victimPageNum, victimEntry);
                
                // Invalidate TLB entry for victim
                tlb.invalidate(victimPageNum);
                
                // Notify page replacer
                pageReplacer->notifyUnloaded(victimPageNum);
            }
        }
        
        // Load page data into physical memory
        std::vector<Byte> pageData(PAGE_SIZE, 0);
        
        if (entry.inBackingStore) {
            // Load from backing store
            pageData = backingStore.readPage(entry.backingStoreLocation);
        }
        // else: New page initialized to zeros
        
        physicalMemory.writePage(frameNum, pageData);
        
        // Update page table entry
        entry.valid = true;
        entry.frameNumber = frameNum;
        entry.referenced = true;
        entry.lastAccessed = std::chrono::steady_clock::now();
        pageTable.updateEntry(pageNum, entry);
        
        // Update TLB
        tlb.update(pageNum, frameNum);
        
        // Notify page replacer
        pageReplacer->notifyLoaded(pageNum);
    }
    
    // Allocate a block of virtual memory
    VirtualAddress allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(managerMutex);
        
        // Round up to whole pages
        size_t numPages = (bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        
        // Find contiguous virtual pages
        size_t foundPages = 0;
        PageNumber startPage = 0;
        
        for (PageNumber i = 0; i < pageTable.size(); ++i) {
            PageTableEntry entry = pageTable.getEntry(i);
            if (!entry.valid && !entry.inBackingStore) {
                if (foundPages == 0) {
                    startPage = i;
                }
                foundPages++;
                
                if (foundPages >= numPages) {
                    break;
                }
            } else {
                foundPages = 0;
            }
        }
        
        if (foundPages < numPages) {
            throw std::runtime_error("Not enough contiguous virtual memory");
        }
        
        // Mark pages as allocated
        for (PageNumber i = startPage; i < startPage + numPages; ++i) {
            PageTableEntry entry;
            entry.valid = false;
            entry.dirty = false;
            entry.referenced = false;
            entry.inBackingStore = false;
            pageTable.updateEntry(i, entry);
        }
        
        return startPage * PAGE_SIZE;
    }
    
    // Free a block of virtual memory
    void free(VirtualAddress addr, size_t bytes) {
        std::lock_guard<std::mutex> lock(managerMutex);
        
        PageNumber startPage = getPageNumber(addr);
        size_t numPages = (bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        
        for (PageNumber i = startPage; i < startPage + numPages; ++i) {
            PageTableEntry entry = pageTable.getEntry(i);
            
            // If page is in memory, free the frame
            if (entry.valid) {
                physicalMemory.freeFrame(entry.frameNumber);
            }
            
            // If page is in backing store, free the block
            if (entry.inBackingStore) {
                backingStore.freeBlock(entry.backingStoreLocation);
            }
            
            // Reset page table entry
            entry = PageTableEntry();
            pageTable.updateEntry(i, entry);
            
            // Invalidate TLB entry
            tlb.invalidate(i);
            
            // Notify page replacer
            pageReplacer->notifyUnloaded(i);
        }
    }
    
    // Get statistics
    struct Statistics {
        size_t pageHits;
        size_t pageFaults;
        size_t tlbHits;
        size_t tlbMisses;
        double pageFaultRate;
        double tlbHitRate;
    };
    
    Statistics getStatistics() const {
        Statistics stats;
        stats.pageHits = pageHits;
        stats.pageFaults = pageFaults;
        stats.tlbHits = tlbHits;
        stats.tlbMisses = tlbMisses;
        
        size_t totalPageAccesses = pageHits + pageFaults;
        stats.pageFaultRate = totalPageAccesses > 0 ? 
            static_cast<double>(pageFaults) / totalPageAccesses : 0.0;
            
        size_t totalTlbLookups = tlbHits + tlbMisses;
        stats.tlbHitRate = totalTlbLookups > 0 ? 
            static_cast<double>(tlbHits) / totalTlbLookups : 0.0;
            
        return stats;
    }
};

//=============================================================================
// Test Functions
//=============================================================================

void testSequentialAccess(VirtualMemoryManager& vmm) {
    std::cout << "Testing sequential access pattern..." << std::endl;
    
    // Allocate memory
    const size_t testSize = PAGE_SIZE * 10;  // 10 pages
    VirtualAddress addr = vmm.allocate(testSize);
    
    // Write sequential data
    for (size_t i = 0; i < testSize; ++i) {
        vmm.writeByte(addr + i, static_cast<Byte>(i % 256));
    }
    
    // Read and verify
    bool dataCorrect = true;
    for (size_t i = 0; i < testSize; ++i) {
        Byte value = vmm.readByte(addr + i);
        if (value != static_cast<Byte>(i % 256)) {
            dataCorrect = false;
            std::cout << "Data mismatch at offset " << i 
                      << ": expected " << static_cast<int>(i % 256) 
                      << ", got " << static_cast<int>(value) << std::endl;
            break;
        }
    }
    
    // Print statistics
    auto stats = vmm.getStatistics();
    std::cout << "Sequential access results:" << std::endl;
    std::cout << "  Data verification: " << (dataCorrect ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Page hits: " << stats.pageHits << std::endl;
    std::cout << "  Page faults: " << stats.pageFaults << std::endl;
    std::cout << "  Page fault rate: " << (stats.pageFaultRate * 100) << "%" << std::endl;
    std::cout << "  TLB hits: " << stats.tlbHits << std::endl;
    std::cout << "  TLB misses: " << stats.tlbMisses << std::endl;
    std::cout << "  TLB hit rate: " << (stats.tlbHitRate * 100) << "%" << std::endl;
    
    // Free memory
    vmm.free(addr, testSize);
}

void testRandomAccess(VirtualMemoryManager& vmm) {
    std::cout << "\nTesting random access pattern..." << std::endl;
    
    // Allocate memory
    const size_t testSize = PAGE_SIZE * 20;  // 20 pages
    VirtualAddress addr = vmm.allocate(testSize);
    
    // Create random access pattern
    std::vector<size_t> accessPattern(testSize);
    for (size_t i = 0; i < testSize; ++i) {
        accessPattern[i] = i;
    }
    
    // Shuffle access pattern
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(accessPattern.begin(), accessPattern.end(), g);
    
    // Write data randomly
    for (size_t i = 0; i < testSize; ++i) {
        size_t offset = accessPattern[i];
        vmm.writeByte(addr + offset, static_cast<Byte>(offset % 256));
    }
    
    // Read and verify randomly
    bool dataCorrect = true;
    for (size_t i = 0; i < testSize; ++i) {
        size_t offset = accessPattern[i];
        Byte value = vmm.readByte(addr + offset);
        if (value != static_cast<Byte>(offset % 256)) {
            dataCorrect = false;
            std::cout << "Data mismatch at offset " << offset 
                      << ": expected " << static_cast<int>(offset % 256) 
                      << ", got " << static_cast<int>(value) << std::endl;
            break;
        }
    }
    
    // Print statistics
    auto stats = vmm.getStatistics();
    std::cout << "Random access results:" << std::endl;
    std::cout << "  Data verification: " << (dataCorrect ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Page hits: " << stats.pageHits << std::endl;
    std::cout << "  Page faults: " << stats.pageFaults << std::endl;
    std::cout << "  Page fault rate: " << (stats.pageFaultRate * 100) << "%" << std::endl;
    std::cout << "  TLB hits: " << stats.tlbHits << std::endl;
    std::cout << "  TLB misses: " << stats.tlbMisses << std::endl;
    std::cout << "  TLB hit rate: " << (stats.tlbHitRate * 100) << "%" << std::endl;
    
    // Free memory
    vmm.free(addr, testSize);
}

void testLocalityOfReference(VirtualMemoryManager& vmm) {
    std::cout << "\nTesting locality of reference..." << std::endl;
    
    // Allocate memory
    const size_t testSize = PAGE_SIZE * 30;  // 30 pages
    VirtualAddress addr = vmm.allocate(testSize);
    
    // Simulate workload with strong locality of reference
    // We'll access 5 "hot" pages 90% of the time, and the rest 10% of the time
    const size_t numAccesses = 10000;
    const size_t hotPages = 5;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Distribution for selecting hot vs. cold access
    std::bernoulli_distribution hotDist(0.9);  // 90% hot, 10% cold
    
    // Distribution for selecting which hot page to access
    std::uniform_int_distribution<> hotPageDist(0, hotPages - 1);
    
    // Distribution for selecting which cold page to access
    std::uniform_int_distribution<> coldPageDist(hotPages, (testSize / PAGE_SIZE) - 1);
    
    // Distribution for offset within a page
    std::uniform_int_distribution<> offsetDist(0, PAGE_SIZE - 1);
    
    // Write data with locality pattern
    for (size_t i = 0; i < numAccesses; ++i) {
        size_t pageIndex;
        
        if (hotDist(gen)) {
            // Access a hot page
            pageIndex = hotPageDist(gen);
        } else {
            // Access a cold page
            pageIndex = coldPageDist(gen);
        }
        
        size_t offset = offsetDist(gen);
        VirtualAddress accessAddr = addr + (pageIndex * PAGE_SIZE) + offset;
        
        // Write a unique value so we can verify it later
        Byte value = static_cast<Byte>((pageIndex * PAGE_SIZE + offset) % 256);
        vmm.writeByte(accessAddr, value);
        
        // Immediately read back to verify (tests TLB)
        Byte readValue = vmm.readByte(accessAddr);
        if (readValue != value) {
            std::cout << "Data verification failed during locality test!" << std::endl;
            break;
        }
    }
    
    // Print statistics
    auto stats = vmm.getStatistics();
    std::cout << "Locality of reference results:" << std::endl;
    std::cout << "  Page hits: " << stats.pageHits << std::endl;
    std::cout << "  Page faults: " << stats.pageFaults << std::endl;
    std::cout << "  Page fault rate: " << (stats.pageFaultRate * 100) << "%" << std::endl;
    std::cout << "  TLB hits: " << stats.tlbHits << std::endl;
    std::cout << "  TLB misses: " << stats.tlbMisses << std::endl;
    std::cout << "  TLB hit rate: " << (stats.tlbHitRate * 100) << "%" << std::endl;
    
    // Free memory
    vmm.free(addr, testSize);
}

//=============================================================================
// Main Function
//=============================================================================

int main() {
    try {
        // Create VirtualMemoryManager with FIFO page replacement
        std::cout << "===== Testing with FIFO Page Replacement =====" << std::endl;
        {
            auto fifoReplacer = std::make_unique<FIFOReplacer>();
            VirtualMemoryManager vmm(
                VIRTUAL_MEMORY_SIZE / PAGE_SIZE,  // Number of virtual pages
                PHYSICAL_MEMORY_SIZE,             // Physical memory size
                BACKING_STORE_SIZE,               // Backing store size
                TLB_SIZE,                         // TLB size
                std::move(fifoReplacer)           // Page replacer
            );
            
            testSequentialAccess(vmm);
            testRandomAccess(vmm);
            testLocalityOfReference(vmm);
        }
        
        // Create VirtualMemoryManager with LRU page replacement
        std::cout << "\n===== Testing with LRU Page Replacement =====" << std::endl;
        {
            auto lruReplacer = std::make_unique<LRUReplacer>();
            VirtualMemoryManager vmm(
                VIRTUAL_MEMORY_SIZE / PAGE_SIZE,  // Number of virtual pages
                PHYSICAL_MEMORY_SIZE,             // Physical memory size
                BACKING_STORE_SIZE,               // Backing store size
                TLB_SIZE,                         // TLB size
                std::move(lruReplacer)            // Page replacer
            );
            
            testSequentialAccess(vmm);
            testRandomAccess(vmm);
            testLocalityOfReference(vmm);
        }
        
        // Create VirtualMemoryManager with Clock page replacement
        std::cout << "\n===== Testing with Clock Page Replacement =====" << std::endl;
        {
            // Use physical memory size / page size for the clock size
            auto clockReplacer = std::make_unique<ClockReplacer>(PHYSICAL_MEMORY_SIZE / PAGE_SIZE);
            VirtualMemoryManager vmm(
                VIRTUAL_MEMORY_SIZE / PAGE_SIZE,  // Number of virtual pages
                PHYSICAL_MEMORY_SIZE,             // Physical memory size
                BACKING_STORE_SIZE,               // Backing store size
                TLB_SIZE,                         // TLB size
                std::move(clockReplacer)          // Page replacer
            );
            
            testSequentialAccess(vmm);
            testRandomAccess(vmm);
            testLocalityOfReference(vmm);
        }
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}