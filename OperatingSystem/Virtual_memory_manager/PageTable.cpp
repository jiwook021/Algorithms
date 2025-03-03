/**
 * @file PageTable.cpp
 * @brief Implementation of PageTable.
 */

#include "PageTable.hpp"

// ---------------------------------------------------------------------------
// PageTable
// ---------------------------------------------------------------------------

PageTable::PageTable(std::size_t num_pages) : entries_(num_pages) {}

PageTableEntry PageTable::GetEntry(std::uint32_t page_num) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page_num >= entries_.size()) {
        throw std::out_of_range("Page number out of range");
    }
    return entries_[page_num];
}

void PageTable::UpdateEntry(std::uint32_t page_num, const PageTableEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page_num >= entries_.size()) {
        throw std::out_of_range("Page number out of range");
    }
    entries_[page_num] = entry;
}

void PageTable::MarkAccessed(std::uint32_t page_num) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page_num >= entries_.size()) {
        throw std::out_of_range("Page number out of range");
    }
    entries_[page_num].referenced   = true;
    entries_[page_num].last_accessed = std::chrono::steady_clock::now();
}

void PageTable::MarkDirty(std::uint32_t page_num) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page_num >= entries_.size()) {
        throw std::out_of_range("Page number out of range");
    }
    entries_[page_num].dirty = true;
}

std::size_t PageTable::Size() const {
    return entries_.size();
}
