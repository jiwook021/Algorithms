/**
 * @file test_Cache.cpp
 * @brief Unit tests for LruCache
 */

#include "Cache.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <string>

namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
} // namespace

TEST(LruCache, BasicPutGet) {
    PrintSection("BasicPutGet",
                 "Verify the fundamental contract: Put stores a value, Get retrieves it.");
    LruCache<int, int> cache(2);
    cache.Put(1, 10);
    cache.Put(2, 20);
    std::cout << "[Result] Get(1) = " << cache.Get(1) << " (expect 10)" << std::endl;
    EXPECT_EQ(cache.Get(1), 10);
    std::cout << "[Result] Get(2) = " << cache.Get(2) << " (expect 20)" << std::endl;
    EXPECT_EQ(cache.Get(2), 20);
}

TEST(LruCache, EvictsLeastRecentlyUsed) {
    PrintSection("EvictsLeastRecentlyUsed",
                 "When capacity is full, the least-recently-used entry must be evicted first.");
    LruCache<int, int> cache(2);
    cache.Put(1, 10);
    cache.Put(2, 20);
    cache.Put(3, 30); // Should evict key 1
    std::cout << "[Result] Exists(1) = " << cache.Exists(1) << " (expect 0 — evicted)" << std::endl;
    EXPECT_FALSE(cache.Exists(1));
    std::cout << "[Result] Exists(2) = " << cache.Exists(2) << " (expect 1)" << std::endl;
    EXPECT_TRUE(cache.Exists(2));
    std::cout << "[Result] Exists(3) = " << cache.Exists(3) << " (expect 1)" << std::endl;
    EXPECT_TRUE(cache.Exists(3));
}

TEST(LruCache, MissThrowsOutOfRange) {
    PrintSection("MissThrowsOutOfRange",
                 "Accessing a key that was never inserted must throw std::out_of_range.");
    LruCache<int, int> cache(2);
    std::cout << "[Result] Get(99) throws std::out_of_range (expect yes)" << std::endl;
    EXPECT_THROW(cache.Get(99), std::out_of_range);
}

TEST(LruCache, GetPromotesEntry) {
    PrintSection("GetPromotesEntry",
                 "A Get() call moves the entry to most-recently-used, protecting it from eviction.");
    LruCache<int, int> cache(2);
    cache.Put(1, 10);
    cache.Put(2, 20);
    cache.Get(1);        // Promote key 1
    cache.Put(3, 30);    // Should evict key 2 (least recent)
    std::cout << "[Result] Exists(2) = " << cache.Exists(2) << " (expect 0 — evicted)" << std::endl;
    EXPECT_FALSE(cache.Exists(2));
    std::cout << "[Result] Exists(1) = " << cache.Exists(1) << " (expect 1 — promoted)" << std::endl;
    EXPECT_TRUE(cache.Exists(1));
    std::cout << "[Result] Exists(3) = " << cache.Exists(3) << " (expect 1)" << std::endl;
    EXPECT_TRUE(cache.Exists(3));
}

TEST(LruCache, UpdateExistingKey) {
    PrintSection("UpdateExistingKey",
                 "Putting the same key again must overwrite the value, not increase size.");
    LruCache<int, int> cache(2);
    cache.Put(1, 10);
    cache.Put(1, 100);
    std::cout << "[Result] Get(1) = " << cache.Get(1) << " (expect 100 — updated)" << std::endl;
    EXPECT_EQ(cache.Get(1), 100);
    std::cout << "[Result] Size() = " << cache.Size() << " (expect 1)" << std::endl;
    EXPECT_EQ(cache.Size(), 1u);
}

TEST(LruCache, SizeTracking) {
    PrintSection("SizeTracking",
                 "Size must grow with inserts and cap at capacity after eviction.");
    LruCache<int, int> cache(3);
    std::cout << "[Result] Size() = " << cache.Size() << " (expect 0 — empty)" << std::endl;
    EXPECT_EQ(cache.Size(), 0u);
    cache.Put(1, 10);
    std::cout << "[Result] Size() = " << cache.Size() << " (expect 1)" << std::endl;
    EXPECT_EQ(cache.Size(), 1u);
    cache.Put(2, 20);
    cache.Put(3, 30);
    std::cout << "[Result] Size() = " << cache.Size() << " (expect 3 — at capacity)" << std::endl;
    EXPECT_EQ(cache.Size(), 3u);
    cache.Put(4, 40);
    std::cout << "[Result] Size() = " << cache.Size() << " (expect 3 — still at capacity)" << std::endl;
    EXPECT_EQ(cache.Size(), 3u);
}

TEST(LruCache, StringKeys) {
    PrintSection("StringKeys",
                 "LruCache works with std::string keys (any hashable type is valid).");
    LruCache<std::string, int> cache(2);
    cache.Put("alpha", 1);
    cache.Put("beta", 2);
    std::cout << "[Result] Get(\"alpha\") = " << cache.Get("alpha") << " (expect 1)" << std::endl;
    EXPECT_EQ(cache.Get("alpha"), 1);
    cache.Put("gamma", 3);
    std::cout << "[Result] Exists(\"beta\") = " << cache.Exists("beta") << " (expect 0 — evicted)" << std::endl;
    EXPECT_FALSE(cache.Exists("beta"));
}

TEST(LruCache, ZeroCapacityThrows) {
    PrintSection("ZeroCapacityThrows",
                 "A zero-capacity cache is nonsensical; the constructor must reject it.");
    std::cout << "[Result] LruCache(0) throws std::invalid_argument (expect yes)" << std::endl;
    using IntCache = LruCache<int, int>;
    EXPECT_THROW(IntCache(0), std::invalid_argument);
}
