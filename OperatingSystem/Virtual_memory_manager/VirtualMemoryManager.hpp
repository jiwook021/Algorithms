/**
 * @file VirtualMemoryManager.hpp
 * @brief Top-level include for the virtual memory subsystem.
 *
 * Pulls together PageTable, TLB, PhysicalMemory, SwapSpace, page-replacement
 * policies (FIFO / LRU / Clock), and the VirtualMemoryManager facade.
 *
 * constexpr configuration constants live here so every translation unit
 * shares a single source of truth.
 */

#pragma once

#include "PageTable.hpp"
#include "TLB.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Configuration constants
// ============================================================================

static constexpr std::size_t kPageSize           = 4096;            // 4 KB pages
static constexpr std::size_t kVirtualMemorySize  = 1UL << 32;      // 4 GB
static constexpr std::size_t kPhysicalMemorySize = 1UL << 24;      // 16 MB
static constexpr std::size_t kTlbSize            = 16;
static constexpr std::size_t kSwapSize           = 1UL << 28;      // 256 MB

// ============================================================================
// Address-translation helpers
// ============================================================================

inline std::uint32_t GetPageNumber(std::uint32_t addr) {
    return addr / kPageSize;
}

inline std::size_t GetOffset(std::uint32_t addr) {
    return addr % kPageSize;
}

inline std::uint32_t MakePhysicalAddress(std::uint32_t frame, std::size_t offset) {
    return frame * kPageSize + offset;
}

// ============================================================================
// Physical Memory Simulation
// ============================================================================

class PhysicalMemory {
public:
    explicit PhysicalMemory(std::size_t size)
        : memory_(size, 0),
          frame_allocated_(size / kPageSize, false),
          num_frames_(size / kPageSize) {}

    /// Allocate a free frame.  Returns true on success.
    bool AllocateFrame(std::uint32_t& frame_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < num_frames_; ++i) {
            if (!frame_allocated_[i]) {
                frame_allocated_[i] = true;
                frame_num = static_cast<std::uint32_t>(i);
                return true;
            }
        }
        return false;
    }

    void FreeFrame(std::uint32_t frame_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_num >= num_frames_) {
            throw std::out_of_range("Frame number out of range");
        }
        frame_allocated_[frame_num] = false;
    }

    std::uint8_t ReadByte(std::uint32_t addr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (addr >= memory_.size()) {
            throw std::out_of_range("Physical address out of range");
        }
        return memory_[addr];
    }

    void WriteByte(std::uint32_t addr, std::uint8_t value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (addr >= memory_.size()) {
            throw std::out_of_range("Physical address out of range");
        }
        memory_[addr] = value;
    }

    void ReadPage(std::uint32_t frame_num, std::vector<std::uint8_t>& buffer) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_num >= num_frames_) {
            throw std::out_of_range("Frame number out of range");
        }
        std::size_t offset = frame_num * kPageSize;
        buffer.resize(kPageSize);
        std::copy(memory_.begin() + static_cast<long>(offset),
                  memory_.begin() + static_cast<long>(offset + kPageSize),
                  buffer.begin());
    }

    void WritePage(std::uint32_t frame_num, const std::vector<std::uint8_t>& buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_num >= num_frames_) {
            throw std::out_of_range("Frame number out of range");
        }
        if (buffer.size() != kPageSize) {
            throw std::invalid_argument("Buffer size must match page size");
        }
        std::size_t offset = frame_num * kPageSize;
        std::copy(buffer.begin(), buffer.end(),
                  memory_.begin() + static_cast<long>(offset));
    }

    std::size_t GetTotalFrames() const { return num_frames_; }

    std::size_t GetFreeFrameCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<std::size_t>(
            std::count(frame_allocated_.begin(), frame_allocated_.end(), false));
    }

private:
    std::vector<std::uint8_t>  memory_;
    std::vector<bool>  frame_allocated_;
    std::size_t        num_frames_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Swap Space (Simulated Disk / Swap Partition)
// ============================================================================

class SwapSpace {
public:
    explicit SwapSpace(std::size_t size)
        : storage_(size, 0),
          slot_allocated_(size / kPageSize, false),
          num_slots_(size / kPageSize) {}

    bool AllocateSlot(std::size_t& slot) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < num_slots_; ++i) {
            if (!slot_allocated_[i]) {
                slot_allocated_[i] = true;
                slot = i;
                return true;
            }
        }
        return false;
    }

    void FreeSlot(std::size_t slot) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= num_slots_) {
            throw std::out_of_range("Swap slot out of range");
        }
        slot_allocated_[slot] = false;
    }

    std::vector<std::uint8_t> ReadPage(std::size_t slot) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= num_slots_) {
            throw std::out_of_range("Swap slot out of range");
        }
        std::vector<std::uint8_t> buffer(kPageSize);
        std::size_t offset = slot * kPageSize;
        std::copy(storage_.begin() + static_cast<long>(offset),
                  storage_.begin() + static_cast<long>(offset + kPageSize),
                  buffer.begin());
        return buffer;
    }

    void WritePage(std::size_t slot, const std::vector<std::uint8_t>& buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= num_slots_) {
            throw std::out_of_range("Swap slot out of range");
        }
        if (buffer.size() != kPageSize) {
            throw std::invalid_argument("Buffer size must match page size");
        }
        std::size_t offset = slot * kPageSize;
        std::copy(buffer.begin(), buffer.end(),
                  storage_.begin() + static_cast<long>(offset));
    }

    std::size_t GetTotalSlots() const { return num_slots_; }

    std::size_t GetFreeSlotCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<std::size_t>(
            std::count(slot_allocated_.begin(), slot_allocated_.end(), false));
    }

private:
    std::vector<std::uint8_t>  storage_;
    std::vector<bool>  slot_allocated_;
    std::size_t        num_slots_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Page Replacement Algorithms
// ============================================================================

/// Abstract interface for page-replacement policies.
class PageReplacer {
public:
    virtual ~PageReplacer() = default;

    virtual std::uint32_t Evict()                    = 0;
    virtual void       NotifyAccessed(std::uint32_t page)   = 0;
    virtual void       NotifyLoaded(std::uint32_t page)     = 0;
    virtual void       NotifyUnloaded(std::uint32_t page)   = 0;
};

// ---------------------------------------------------------------------------
// FIFO
// ---------------------------------------------------------------------------

class FIFOReplacer : public PageReplacer {
public:
    std::uint32_t Evict() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        std::uint32_t victim = queue_.front();
        queue_.pop();
        set_.erase(victim);
        return victim;
    }

    void NotifyAccessed(std::uint32_t /*page*/) override { /* FIFO ignores accesses */ }

    void NotifyLoaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (set_.find(page) == set_.end()) {
            queue_.push(page);
            set_.insert(page);
        }
    }

    void NotifyUnloaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        set_.erase(page);
    }

private:
    std::queue<std::uint32_t>              queue_;
    std::unordered_set<std::uint32_t>      set_;
    mutable std::mutex                  mutex_;
};

// ---------------------------------------------------------------------------
// LRU
// ---------------------------------------------------------------------------

class LRUReplacer : public PageReplacer {
public:
    std::uint32_t Evict() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lru_list_.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        std::uint32_t victim = lru_list_.back();
        lru_list_.pop_back();
        map_.erase(victim);
        return victim;
    }

    void NotifyAccessed(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(page);
        if (it != map_.end()) {
            lru_list_.erase(it->second);
            lru_list_.push_front(page);
            it->second = lru_list_.begin();
        }
    }

    void NotifyLoaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (map_.find(page) == map_.end()) {
            lru_list_.push_front(page);
            map_[page] = lru_list_.begin();
        }
    }

    void NotifyUnloaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(page);
        if (it != map_.end()) {
            lru_list_.erase(it->second);
            map_.erase(it);
        }
    }

private:
    std::unordered_map<std::uint32_t, std::list<std::uint32_t>::iterator> map_;
    mutable std::list<std::uint32_t>                                   lru_list_;
    mutable std::mutex                                              mutex_;
};

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

class ClockReplacer : public PageReplacer {
public:
    explicit ClockReplacer(std::size_t max_pages)
        : clock_buffer_(max_pages, 0),
          reference_bit_(max_pages, false),
          clock_hand_(0) {}

    std::uint32_t Evict() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (page_to_pos_.empty()) {
            throw std::runtime_error("No pages to replace");
        }
        while (true) {
            if (!reference_bit_[clock_hand_]) {
                std::uint32_t victim = clock_buffer_[clock_hand_];
                page_to_pos_.erase(victim);
                clock_hand_ = (clock_hand_ + 1) % clock_buffer_.size();
                return victim;
            }
            reference_bit_[clock_hand_] = false;
            clock_hand_ = (clock_hand_ + 1) % clock_buffer_.size();

            if (page_to_pos_.size() < clock_buffer_.size() &&
                std::all_of(reference_bit_.begin(), reference_bit_.end(),
                            [](bool b) { return b; })) {
                std::uint32_t victim = clock_buffer_[clock_hand_];
                page_to_pos_.erase(victim);
                clock_hand_ = (clock_hand_ + 1) % clock_buffer_.size();
                return victim;
            }
        }
    }

    void NotifyAccessed(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = page_to_pos_.find(page);
        if (it != page_to_pos_.end()) {
            reference_bit_[it->second] = true;
        }
    }

    void NotifyLoaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (page_to_pos_.find(page) == page_to_pos_.end()) {
            std::size_t pos;
            if (page_to_pos_.size() < clock_buffer_.size()) {
                pos = page_to_pos_.size();
            } else {
                std::uint32_t victim = Evict();
                pos = page_to_pos_[victim];
                page_to_pos_.erase(victim);
            }
            clock_buffer_[pos]  = page;
            reference_bit_[pos] = true;
            page_to_pos_[page]  = pos;
        }
    }

    void NotifyUnloaded(std::uint32_t page) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = page_to_pos_.find(page);
        if (it != page_to_pos_.end()) {
            reference_bit_[it->second] = false;
        }
    }

private:
    std::unordered_map<std::uint32_t, std::size_t> page_to_pos_;
    std::vector<std::uint32_t>                     clock_buffer_;
    std::vector<bool>                           reference_bit_;
    std::size_t                                 clock_hand_;
    mutable std::mutex                          mutex_;
};

// ============================================================================
// VirtualMemoryManager  (facade)
// ============================================================================

class VirtualMemoryManager {
public:
    /// Runtime-selectable replacement policy.
    enum class ReplacementPolicy { FIFO, LRU, Clock };

    /// Configuration constants re-exported for callers.
    static constexpr std::size_t kPageSize           = ::kPageSize;
    static constexpr std::size_t kPhysicalMemorySize = ::kPhysicalMemorySize;
    static constexpr std::size_t kSwapSize           = ::kSwapSize;

    /// Construct with explicit sizes and a pre-built replacer.
    VirtualMemoryManager(std::size_t num_virtual_pages,
                         std::size_t physical_memory_size,
                         std::size_t swap_size,
                         std::size_t tlb_size,
                         std::unique_ptr<PageReplacer> replacer)
        : page_table_(num_virtual_pages),
          tlb_(tlb_size),
          physical_memory_(physical_memory_size),
          swap_(swap_size),
          replacer_(std::move(replacer)) {}

    // ----- Core operations --------------------------------------------------

    /// Translate a virtual address to a physical address.
    std::uint32_t Translate(std::uint32_t addr) {
        std::uint32_t  page   = GetPageNumber(addr);
        std::size_t offset = GetOffset(addr);

        // TLB probe
        auto frame_opt = tlb_.Lookup(page);
        if (frame_opt.has_value()) {
            replacer_->NotifyAccessed(page);
            ++page_hits_;
            return MakePhysicalAddress(*frame_opt, offset);
        }

        // Page table lookup
        PageTableEntry entry = page_table_.GetEntry(page);
        if (entry.valid) {
            ++page_hits_;
            replacer_->NotifyAccessed(page);
            tlb_.Insert(page, static_cast<std::uint32_t>(entry.frame_number));
            return MakePhysicalAddress(static_cast<std::uint32_t>(entry.frame_number), offset);
        }

        // Page fault
        ++page_faults_;
        HandlePageFault(page);
        return Translate(addr);   // retry
    }

    /// Read a single byte from virtual memory.
    std::uint8_t ReadByte(std::uint32_t addr) {
        std::uint32_t phys = Translate(addr);
        return physical_memory_.ReadByte(phys);
    }

    /// Write a single byte to virtual memory.
    void WriteByte(std::uint32_t addr, std::uint8_t value) {
        std::uint32_t page = GetPageNumber(addr);

        std::uint32_t phys = Translate(addr);

        // Mark dirty in page table
        PageTableEntry entry = page_table_.GetEntry(page);
        entry.dirty = true;
        page_table_.UpdateEntry(page, entry);

        physical_memory_.WriteByte(phys, value);
    }

    /// Allocate contiguous virtual memory (returns starting virtual address).
    std::uint32_t Allocate(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        std::size_t num_pages = (bytes + kPageSize - 1) / kPageSize;
        std::size_t found     = 0;
        std::uint32_t  start     = 0;

        for (std::uint32_t i = 0; i < page_table_.Size(); ++i) {
            PageTableEntry e = page_table_.GetEntry(i);
            if (!e.valid && !e.swapped) {
                if (found == 0) start = i;
                ++found;
                if (found >= num_pages) break;
            } else {
                found = 0;
            }
        }

        if (found < num_pages) {
            throw std::runtime_error("Not enough contiguous virtual memory");
        }

        for (std::uint32_t i = start; i < start + num_pages; ++i) {
            PageTableEntry e{};
            page_table_.UpdateEntry(i, e);
        }

        return start * kPageSize;
    }

    /// Free a previously allocated virtual memory region.
    void Free(std::uint32_t addr, std::size_t bytes) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        std::uint32_t  start     = GetPageNumber(addr);
        std::size_t num_pages = (bytes + kPageSize - 1) / kPageSize;

        for (std::uint32_t i = start; i < start + num_pages; ++i) {
            PageTableEntry e = page_table_.GetEntry(i);

            if (e.valid) {
                physical_memory_.FreeFrame(static_cast<std::uint32_t>(e.frame_number));
            }
            if (e.swapped) {
                swap_.FreeSlot(e.swap_offset);
            }

            page_table_.UpdateEntry(i, PageTableEntry{});
            tlb_.Invalidate(i);
            replacer_->NotifyUnloaded(i);
        }
    }

    // ----- Statistics -------------------------------------------------------

    struct Statistics {
        std::size_t page_hits;
        std::size_t page_faults;
        std::size_t tlb_hits;
        std::size_t tlb_misses;
        double      page_fault_rate;
        double      tlb_hit_rate;
    };

    Statistics GetStatistics() const {
        Statistics s{};
        s.page_hits   = page_hits_;
        s.page_faults = page_faults_;
        s.tlb_hits    = tlb_.Hits();
        s.tlb_misses  = tlb_.Misses();

        std::size_t total_page = s.page_hits + s.page_faults;
        s.page_fault_rate = total_page > 0
            ? static_cast<double>(s.page_faults) / static_cast<double>(total_page)
            : 0.0;

        s.tlb_hit_rate = tlb_.HitRate();
        return s;
    }

private:
    void HandlePageFault(std::uint32_t page) {
        std::lock_guard<std::mutex> lock(manager_mutex_);

        PageTableEntry entry = page_table_.GetEntry(page);

        std::uint32_t frame_num;
        bool allocated = physical_memory_.AllocateFrame(frame_num);

        if (!allocated) {
            // Evict a victim page
            std::uint32_t victim = replacer_->Evict();
            PageTableEntry victim_entry = page_table_.GetEntry(victim);
            frame_num = static_cast<std::uint32_t>(victim_entry.frame_number);

            if (victim_entry.dirty) {
                std::vector<std::uint8_t> page_data;
                physical_memory_.ReadPage(frame_num, page_data);

                std::size_t swap_slot;
                if (victim_entry.swapped) {
                    swap_slot = victim_entry.swap_offset;
                } else {
                    if (!swap_.AllocateSlot(swap_slot)) {
                        throw std::runtime_error("Out of swap space");
                    }
                }
                swap_.WritePage(swap_slot, page_data);

                victim_entry.valid             = false;
                victim_entry.swapped  = true;
                victim_entry.swap_offset = swap_slot;
            } else {
                victim_entry.valid = false;
            }

            page_table_.UpdateEntry(victim, victim_entry);
            tlb_.Invalidate(victim);
            replacer_->NotifyUnloaded(victim);
        }

        // Load page data
        std::vector<std::uint8_t> page_data(kPageSize, 0);
        if (entry.swapped) {
            page_data = swap_.ReadPage(entry.swap_offset);
        }
        physical_memory_.WritePage(frame_num, page_data);

        // Update page table
        entry.valid        = true;
        entry.frame_number = frame_num;
        entry.referenced   = true;
        entry.last_accessed = std::chrono::steady_clock::now();
        page_table_.UpdateEntry(page, entry);

        // Update TLB
        tlb_.Insert(page, frame_num);

        // Notify replacer
        replacer_->NotifyLoaded(page);
    }

    PageTable                      page_table_;
    TLB                            tlb_;
    PhysicalMemory                 physical_memory_;
    SwapSpace                      swap_;
    std::unique_ptr<PageReplacer>  replacer_;

    mutable std::mutex             manager_mutex_;

    std::size_t page_hits_{0};
    std::size_t page_faults_{0};
};

// ============================================================================
// Demo / Test Helper Functions (used by main.cpp)
// ============================================================================

inline void TestSequentialAccess(VirtualMemoryManager& vmm) {
    std::cout << "Testing sequential access pattern..." << std::endl;

    const std::size_t test_size = kPageSize * 10;
    std::uint32_t addr = vmm.Allocate(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        vmm.WriteByte(addr + static_cast<std::uint32_t>(i),
                      static_cast<std::uint8_t>(i % 256));
    }

    bool correct = true;
    for (std::size_t i = 0; i < test_size; ++i) {
        std::uint8_t v = vmm.ReadByte(addr + static_cast<std::uint32_t>(i));
        if (v != static_cast<std::uint8_t>(i % 256)) {
            correct = false;
            std::cout << "Data mismatch at offset " << i
                      << ": expected " << static_cast<int>(i % 256)
                      << ", got " << static_cast<int>(v) << std::endl;
            break;
        }
    }

    auto s = vmm.GetStatistics();
    std::cout << "Sequential access results:\n"
              << "  Data verification: " << (correct ? "PASSED" : "FAILED") << "\n"
              << "  Page hits: "       << s.page_hits << "\n"
              << "  Page faults: "     << s.page_faults << "\n"
              << "  Page fault rate: " << (s.page_fault_rate * 100) << "%\n"
              << "  TLB hits: "        << s.tlb_hits << "\n"
              << "  TLB misses: "      << s.tlb_misses << "\n"
              << "  TLB hit rate: "    << (s.tlb_hit_rate * 100) << "%\n";

    vmm.Free(addr, test_size);
}

inline void TestRandomAccess(VirtualMemoryManager& vmm) {
    std::cout << "\nTesting random access pattern..." << std::endl;

    const std::size_t test_size = kPageSize * 20;
    std::uint32_t addr = vmm.Allocate(test_size);

    std::vector<std::size_t> pattern(test_size);
    for (std::size_t i = 0; i < test_size; ++i) pattern[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pattern.begin(), pattern.end(), gen);

    for (std::size_t i = 0; i < test_size; ++i) {
        std::size_t off = pattern[i];
        vmm.WriteByte(addr + static_cast<std::uint32_t>(off),
                      static_cast<std::uint8_t>(off % 256));
    }

    bool correct = true;
    for (std::size_t i = 0; i < test_size; ++i) {
        std::size_t off = pattern[i];
        std::uint8_t v = vmm.ReadByte(addr + static_cast<std::uint32_t>(off));
        if (v != static_cast<std::uint8_t>(off % 256)) {
            correct = false;
            std::cout << "Data mismatch at offset " << off
                      << ": expected " << static_cast<int>(off % 256)
                      << ", got " << static_cast<int>(v) << std::endl;
            break;
        }
    }

    auto s = vmm.GetStatistics();
    std::cout << "Random access results:\n"
              << "  Data verification: " << (correct ? "PASSED" : "FAILED") << "\n"
              << "  Page hits: "       << s.page_hits << "\n"
              << "  Page faults: "     << s.page_faults << "\n"
              << "  Page fault rate: " << (s.page_fault_rate * 100) << "%\n"
              << "  TLB hits: "        << s.tlb_hits << "\n"
              << "  TLB misses: "      << s.tlb_misses << "\n"
              << "  TLB hit rate: "    << (s.tlb_hit_rate * 100) << "%\n";

    vmm.Free(addr, test_size);
}

inline void TestLocalityOfReference(VirtualMemoryManager& vmm) {
    std::cout << "\nTesting locality of reference..." << std::endl;

    const std::size_t test_size   = kPageSize * 30;
    std::uint32_t addr           = vmm.Allocate(test_size);

    const std::size_t num_accesses = 10000;
    const std::size_t hot_pages    = 5;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::bernoulli_distribution hot_dist(0.9);
    std::uniform_int_distribution<> hot_page_dist(0, static_cast<int>(hot_pages - 1));
    std::uniform_int_distribution<> cold_page_dist(
        static_cast<int>(hot_pages),
        static_cast<int>((test_size / kPageSize) - 1));
    std::uniform_int_distribution<> offset_dist(0, static_cast<int>(kPageSize - 1));

    for (std::size_t i = 0; i < num_accesses; ++i) {
        std::size_t page_idx = hot_dist(gen)
            ? static_cast<std::size_t>(hot_page_dist(gen))
            : static_cast<std::size_t>(cold_page_dist(gen));

        std::size_t off = static_cast<std::size_t>(offset_dist(gen));
        std::uint32_t a = addr
            + static_cast<std::uint32_t>(page_idx * kPageSize + off);

        std::uint8_t val = static_cast<std::uint8_t>((page_idx * kPageSize + off) % 256);
        vmm.WriteByte(a, val);

        std::uint8_t read_val = vmm.ReadByte(a);
        if (read_val != val) {
            std::cout << "Data verification failed during locality test!\n";
            break;
        }
    }

    auto s = vmm.GetStatistics();
    std::cout << "Locality of reference results:\n"
              << "  Page hits: "       << s.page_hits << "\n"
              << "  Page faults: "     << s.page_faults << "\n"
              << "  Page fault rate: " << (s.page_fault_rate * 100) << "%\n"
              << "  TLB hits: "        << s.tlb_hits << "\n"
              << "  TLB misses: "      << s.tlb_misses << "\n"
              << "  TLB hit rate: "    << (s.tlb_hit_rate * 100) << "%\n";

    vmm.Free(addr, test_size);
}
