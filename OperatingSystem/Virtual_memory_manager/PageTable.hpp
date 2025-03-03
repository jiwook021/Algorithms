/**
 * @file PageTable.hpp
 * @brief Page table entry and page table for virtual memory management.
 *
 * Provides the PageTableEntry struct (valid/dirty/referenced bits, frame mapping,
 * swap metadata) and the PageTable class (thread-safe vector of entries
 * with access/dirty tracking for page-replacement algorithms).
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// PageTableEntry
// ---------------------------------------------------------------------------

/// Single entry in the page table.
struct PageTableEntry {
    bool        valid{false};
    bool        dirty{false};
    bool        referenced{false};
    std::size_t frame_number{0};

    // Swap tracking
    bool        swapped{false};
    std::size_t swap_offset{0};

    // Timestamp for LRU page replacement
    std::chrono::steady_clock::time_point last_accessed{};
};

// ---------------------------------------------------------------------------
// PageTable
// ---------------------------------------------------------------------------

/// Thread-safe page table backed by a flat vector of PageTableEntry.
class PageTable {
public:
    /// Construct a page table with \p num_pages entries (all initially invalid).
    explicit PageTable(std::size_t num_pages);

    /// Return a copy of the entry for \p page_num (thread-safe).
    PageTableEntry GetEntry(std::uint32_t page_num) const;

    /// Overwrite the entry for \p page_num (thread-safe).
    void UpdateEntry(std::uint32_t page_num, const PageTableEntry& entry);

    /// Set the referenced flag and update the LRU timestamp (thread-safe).
    void MarkAccessed(std::uint32_t page_num);

    /// Set the dirty flag (thread-safe).
    void MarkDirty(std::uint32_t page_num);

    /// Return the number of entries.
    std::size_t Size() const;

private:
    std::vector<PageTableEntry> entries_;
    mutable std::mutex          mutex_;
};
