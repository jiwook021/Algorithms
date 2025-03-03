/**
 * @file TLB.cpp
 * @brief Implementation of the Translation Lookaside Buffer.
 */

#include "TLB.hpp"

// ---------------------------------------------------------------------------
// TLB
// ---------------------------------------------------------------------------

TLB::TLB(std::size_t capacity) : capacity_(capacity) {}

std::optional<std::uint32_t> TLB::Lookup(std::uint32_t page) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = map_.find(page);
    if (it != map_.end()) {
        ++hits_;
        std::uint32_t frame = it->second->frame;

        // Move to front (MRU)
        order_.splice(order_.begin(), order_, it->second);
        return frame;
    }

    ++misses_;
    return std::nullopt;
}

void TLB::Insert(std::uint32_t page, std::uint32_t frame) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = map_.find(page);
    if (it != map_.end()) {
        // Update existing entry and promote to MRU
        it->second->frame = frame;
        order_.splice(order_.begin(), order_, it->second);
        return;
    }

    // Evict LRU entry if at capacity
    if (order_.size() >= capacity_) {
        auto& victim = order_.back();
        map_.erase(victim.page);
        order_.pop_back();
    }

    // Insert new entry at front (MRU)
    order_.push_front({page, frame});
    map_[page] = order_.begin();
}

void TLB::Invalidate(std::uint32_t page) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = map_.find(page);
    if (it != map_.end()) {
        order_.erase(it->second);
        map_.erase(it);
    }
}

void TLB::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    order_.clear();
    map_.clear();
}

double TLB::HitRate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t total = hits_ + misses_;
    return total > 0 ? static_cast<double>(hits_) / static_cast<double>(total)
                     : 0.0;
}

std::size_t TLB::Hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hits_;
}

std::size_t TLB::Misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return misses_;
}
