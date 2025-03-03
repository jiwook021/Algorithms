/**
 * @file TLB.hpp
 * @brief Translation Lookaside Buffer with LRU eviction.
 *
 * Caches recent virtual-page-to-physical-frame mappings.  Uses a
 * doubly-linked list (front = MRU) plus a hash map for O(1) lookup
 * and O(1) eviction.  Thread-safe.  Tracks hit / miss counters so
 * callers can query HitRate().
 */

#pragma once

#include "PageTable.hpp"

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

// ---------------------------------------------------------------------------
// TLB
// ---------------------------------------------------------------------------

class TLB {
public:
    /// Construct a TLB with the given \p capacity (number of entries).
    explicit TLB(std::size_t capacity);

    /// Look up \p page in the TLB.
    /// Returns the mapped frame number on hit, or std::nullopt on miss.
    /// Updates LRU order and hit/miss counters.
    std::optional<std::uint32_t> Lookup(std::uint32_t page);

    /// Insert or update the mapping \p page -> \p frame.
    /// Evicts the LRU entry when at capacity.
    void Insert(std::uint32_t page, std::uint32_t frame);

    /// Remove the entry for \p page (e.g., when the page is swapped out).
    void Invalidate(std::uint32_t page);

    /// Remove all entries.
    void Clear();

    /// Return the TLB hit rate (hits / total lookups), or 0 if no lookups.
    double HitRate() const;

    /// Return total hit count.
    std::size_t Hits() const;

    /// Return total miss count.
    std::size_t Misses() const;

private:
    struct Entry {
        std::uint32_t  page;
        std::uint32_t frame;
    };

    std::size_t capacity_;

    // Front = MRU, back = LRU
    std::list<Entry> order_;
    std::unordered_map<std::uint32_t, std::list<Entry>::iterator> map_;

    mutable std::size_t hits_{0};
    mutable std::size_t misses_{0};

    mutable std::mutex mutex_;
};
