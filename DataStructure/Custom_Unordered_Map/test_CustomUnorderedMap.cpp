/**
 * @file test_CustomUnorderedMap.cpp
 * @brief Google Test suite for UnorderedMap with educational PrintSection output.
 *
 * Covers: O(1) insert/lookup, automatic rehashing on load factor, collision
 * handling via separate chaining, configurable bucket count and load factor,
 * erase, copy/move semantics, and iterator traversal.
 */

#include <gtest/gtest.h>
#include "CustomUnorderedMap.hpp"

#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
} // namespace

// ===================================================================
// 1. Basic Types Work
// ===================================================================

TEST(UnorderedMapTest, BasicTypesWork) {
    PrintSection("Basic Types Work",
                 "UnorderedMap works with common key types like int, string, "
                 "and double — any type with a std::hash specialization.");

    UnorderedMap<int, std::string> int_map;
    int_map.Insert(1, "one");
    EXPECT_EQ(int_map.Size(), 1u);

    UnorderedMap<std::string, int> str_map;
    str_map.Insert("hello", 42);
    EXPECT_EQ(str_map.Size(), 1u);

    std::cout << "  Verified: int and string keys both work.\n";
}

// ===================================================================
// 2. Configurable Bucket Count and Load Factor
// ===================================================================

TEST(UnorderedMapTest, ConfigurableBucketCountAndLoadFactor) {
    PrintSection("Configurable Buckets & Load Factor",
                 "The constructor accepts initial_buckets and max_load_factor so "
                 "callers can tune memory vs. collision trade-offs.");

    UnorderedMap<int, int> map_small(4, 0.5);
    EXPECT_EQ(map_small.BucketCount(), 4u);
    EXPECT_DOUBLE_EQ(map_small.MaxLoadFactor(), 0.5);

    UnorderedMap<int, int> map_large(64, 2.0);
    EXPECT_EQ(map_large.BucketCount(), 64u);
    EXPECT_DOUBLE_EQ(map_large.MaxLoadFactor(), 2.0);

    // Default constructor: 16 buckets, load factor 1.0
    UnorderedMap<int, int> map_default;
    EXPECT_EQ(map_default.BucketCount(), 16u);
    EXPECT_DOUBLE_EQ(map_default.MaxLoadFactor(), 1.0);

    std::cout << "  Verified: buckets and load factor are configurable.\n";
}

// ===================================================================
// 3. O(1) Insert and Lookup
// ===================================================================

TEST(UnorderedMapTest, InsertAndLookupAmortizedConstant) {
    PrintSection("O(1) Insert & Lookup",
                 "Hash tables provide amortized O(1) insert and lookup. "
                 "We verify this by timing N vs 2N operations and checking "
                 "the ratio stays roughly constant.");

    constexpr int kSmallN = 50000;
    constexpr int kLargeN = 100000;

    // --- Insert timing ---
    auto InsertAndTime = [](int n) {
        UnorderedMap<int, int> map(static_cast<std::size_t>(n), 2.0);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            map.Insert(i, i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };

    long long small_insert = InsertAndTime(kSmallN);
    long long large_insert = InsertAndTime(kLargeN);

    double insert_ratio = (small_insert > 0)
        ? static_cast<double>(large_insert) / small_insert
        : 0.0;

    std::cout << "  Insert " << kSmallN << ": " << small_insert << " us\n";
    std::cout << "  Insert " << kLargeN << ": " << large_insert << " us\n";
    std::cout << "  Ratio (expect ~2.0 for O(n) total): " << insert_ratio << "\n";

    // Ratio should be roughly 2x (O(n) total => O(1) amortized per op).
    // Allow generous bounds for CI variability.
    EXPECT_LT(insert_ratio, 5.0) << "Insert does not appear amortized O(1)";

    // --- Lookup timing ---
    UnorderedMap<int, int> map(static_cast<std::size_t>(kLargeN), 2.0);
    for (int i = 0; i < kLargeN; ++i) map.Insert(i, i);

    auto LookupTime = [&](int n) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            volatile auto* p = map.Find(i);
            (void)p;
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };

    long long small_lookup = LookupTime(kSmallN);
    long long large_lookup = LookupTime(kLargeN);

    double lookup_ratio = (small_lookup > 0)
        ? static_cast<double>(large_lookup) / small_lookup
        : 0.0;

    std::cout << "  Lookup " << kSmallN << ": " << small_lookup << " us\n";
    std::cout << "  Lookup " << kLargeN << ": " << large_lookup << " us\n";
    std::cout << "  Ratio (expect ~2.0 for O(n) total): " << lookup_ratio << "\n";

    EXPECT_LT(lookup_ratio, 5.0) << "Lookup does not appear amortized O(1)";
}

// ===================================================================
// 4. Automatic Rehashing on Load Factor
// ===================================================================

TEST(UnorderedMapTest, AutomaticRehashingOnLoadFactor) {
    PrintSection("Automatic Rehashing",
                 "When the load factor exceeds max_load_factor, the table "
                 "doubles its bucket count automatically. This keeps chain "
                 "lengths short and preserves O(1) performance.");

    UnorderedMap<int, int> map(4, 1.0);
    EXPECT_EQ(map.BucketCount(), 4u);

    std::size_t initial_buckets = map.BucketCount();

    // Insert enough elements to exceed load factor of 1.0 with 4 buckets
    for (int i = 0; i < 10; ++i) {
        map.Insert(i, i * 10);
    }

    std::cout << "  Initial buckets: " << initial_buckets << "\n";
    std::cout << "  After 10 inserts: " << map.BucketCount() << " buckets\n";
    std::cout << "  Load factor: " << map.LoadFactor() << "\n";

    EXPECT_GT(map.BucketCount(), initial_buckets)
        << "Bucket count should have grown after exceeding load factor";
    EXPECT_LE(map.LoadFactor(), map.MaxLoadFactor() + 0.01)
        << "Load factor should be at or below max after rehash";

    // All elements must still be accessible after rehash
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(map.Contains(i)) << "Key " << i << " lost after rehash";
        auto* val = map.Find(i);
        ASSERT_NE(val, nullptr);
        EXPECT_EQ(*val, i * 10);
    }

    EXPECT_EQ(map.Size(), 10u);
    std::cout << "  All 10 elements verified present after rehash.\n";
}

TEST(UnorderedMapTest, RehashWithCustomLoadFactor) {
    PrintSection("Rehash with Custom Load Factor",
                 "A lower max_load_factor triggers rehashing sooner, "
                 "producing more buckets and fewer collisions.");

    UnorderedMap<int, int> map(4, 0.5);
    for (int i = 0; i < 8; ++i) {
        map.Insert(i, i);
    }

    std::cout << "  Buckets after 8 inserts (max_lf=0.5, init=4): "
              << map.BucketCount() << "\n";
    std::cout << "  Load factor: " << map.LoadFactor() << "\n";

    // With max_load_factor 0.5 and 8 elements, we need at least 16 buckets
    EXPECT_GE(map.BucketCount(), 16u);
    EXPECT_LE(map.LoadFactor(), 0.5 + 0.01);
}

// ===================================================================
// 5. Collision Handling via Separate Chaining
// ===================================================================

TEST(UnorderedMapTest, CollisionHandlingSeparateChaining) {
    PrintSection("Collision Handling - Separate Chaining",
                 "When multiple keys hash to the same bucket, separate "
                 "chaining stores them in a linked list. All keys remain "
                 "accessible despite 100% collision rate.");

    // Force all elements into 1 bucket by using bucket_count=1
    // and a very high max_load_factor to prevent rehashing.
    UnorderedMap<int, std::string> map(1, 100.0);

    map.Insert(1, "one");
    map.Insert(2, "two");
    map.Insert(3, "three");
    map.Insert(4, "four");
    map.Insert(5, "five");

    EXPECT_EQ(map.Size(), 5u);

    // All five must be in bucket 0 (the only bucket)
    EXPECT_EQ(map.BucketSize(0), 5u);

    std::cout << "  All 5 elements in bucket 0 (full collision).\n";

    // Every element must still be individually retrievable
    auto* v1 = map.Find(1);
    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(*v1, "one");

    auto* v3 = map.Find(3);
    ASSERT_NE(v3, nullptr);
    EXPECT_EQ(*v3, "three");

    auto* v5 = map.Find(5);
    ASSERT_NE(v5, nullptr);
    EXPECT_EQ(*v5, "five");

    // Erase from the middle of a collision chain
    EXPECT_TRUE(map.Erase(3));
    EXPECT_FALSE(map.Contains(3));
    EXPECT_EQ(map.BucketSize(0), 4u);

    // Other elements in the chain remain intact
    EXPECT_TRUE(map.Contains(1));
    EXPECT_TRUE(map.Contains(2));
    EXPECT_TRUE(map.Contains(4));
    EXPECT_TRUE(map.Contains(5));

    std::cout << "  Erase from collision chain verified; remaining 4 intact.\n";
}

// ===================================================================
// 6. Insert Duplicate Key
// ===================================================================

TEST(UnorderedMapTest, InsertDuplicateKeyDoesNotOverwrite) {
    PrintSection("Insert Duplicate Key",
                 "Inserting a key that already exists should return false "
                 "and leave the original value unchanged.");

    UnorderedMap<std::string, int> map;

    auto [it1, ok1] = map.Insert("apple", 5);
    EXPECT_TRUE(ok1);
    EXPECT_EQ(map.Size(), 1u);

    auto [it2, ok2] = map.Insert("apple", 99);
    EXPECT_FALSE(ok2);
    EXPECT_EQ(map.Size(), 1u);
    EXPECT_EQ(map.At("apple"), 5); // original value kept
}

// ===================================================================
// 7. Erase
// ===================================================================

TEST(UnorderedMapTest, EraseByKey) {
    PrintSection("Erase by Key",
                 "Erase returns true if the key existed, false otherwise.");

    UnorderedMap<std::string, int> map;
    map.Insert("a", 1);
    map.Insert("b", 2);
    map.Insert("c", 3);

    EXPECT_TRUE(map.Erase("b"));
    EXPECT_EQ(map.Size(), 2u);
    EXPECT_FALSE(map.Contains("b"));

    EXPECT_FALSE(map.Erase("nonexistent"));
    EXPECT_EQ(map.Size(), 2u);
}

// ===================================================================
// 8. Clear
// ===================================================================

TEST(UnorderedMapTest, Clear) {
    PrintSection("Clear",
                 "Clear removes all elements but preserves bucket structure.");

    UnorderedMap<int, int> map;
    for (int i = 0; i < 50; ++i) map.Insert(i, i * 10);
    EXPECT_EQ(map.Size(), 50u);

    std::size_t buckets_before = map.BucketCount();
    map.Clear();
    EXPECT_EQ(map.Size(), 0u);
    EXPECT_TRUE(map.Empty());
    EXPECT_EQ(map.BucketCount(), buckets_before);
}

// ===================================================================
// 9. Operator[] and At
// ===================================================================

TEST(UnorderedMapTest, SubscriptOperator) {
    PrintSection("operator[]",
                 "operator[] inserts a default-constructed value for missing "
                 "keys and returns a mutable reference.");

    UnorderedMap<std::string, int> map;
    map["x"] = 42;
    EXPECT_EQ(map["x"], 42);

    map["x"] = 99;
    EXPECT_EQ(map["x"], 99);

    // Accessing a non-existent key creates a default entry
    int val = map["new_key"];
    EXPECT_EQ(val, 0);
    EXPECT_EQ(map.Size(), 2u);
}

TEST(UnorderedMapTest, AtThrowsOnMissing) {
    PrintSection("At() Bounds Checking",
                 "At() throws std::out_of_range when the key is absent, "
                 "unlike operator[] which silently inserts.");

    UnorderedMap<int, int> map;
    map.Insert(1, 100);
    EXPECT_EQ(map.At(1), 100);
    EXPECT_THROW(map.At(999), std::out_of_range);
}

// ===================================================================
// 10. Iterator Traversal
// ===================================================================

TEST(UnorderedMapTest, IteratorTraversal) {
    PrintSection("Iterator Traversal",
                 "Iterating visits every element exactly once, regardless "
                 "of bucket distribution.");

    UnorderedMap<int, int> map;
    map.Insert(1, 10);
    map.Insert(2, 20);
    map.Insert(3, 30);

    int count = 0;
    int sum = 0;
    for (auto it = map.begin(); it != map.end(); ++it) {
        sum += (*it).second;
        ++count;
    }
    EXPECT_EQ(count, 3);
    EXPECT_EQ(sum, 60);
}

// ===================================================================
// 11. Copy Constructor
// ===================================================================

TEST(UnorderedMapTest, CopyConstructor) {
    PrintSection("Copy Constructor",
                 "A copied map is an independent deep copy; mutations to "
                 "the copy must not affect the original.");

    UnorderedMap<std::string, int> map;
    map.Insert("a", 1);
    map.Insert("b", 2);

    UnorderedMap<std::string, int> copy(map);
    EXPECT_EQ(copy.Size(), 2u);
    EXPECT_EQ(copy.At("a"), 1);
    EXPECT_EQ(copy.At("b"), 2);

    // Mutation independence
    copy.Insert("c", 3);
    EXPECT_EQ(copy.Size(), 3u);
    EXPECT_EQ(map.Size(), 2u);
}

// ===================================================================
// 12. Move Constructor
// ===================================================================

TEST(UnorderedMapTest, MoveConstructor) {
    PrintSection("Move Constructor",
                 "After move, the source map is left empty and the "
                 "destination owns all resources.");

    UnorderedMap<std::string, int> map;
    map.Insert("x", 42);

    UnorderedMap<std::string, int> moved(std::move(map));
    EXPECT_EQ(moved.Size(), 1u);
    EXPECT_EQ(moved.At("x"), 42);
    EXPECT_EQ(map.Size(), 0u);
}

// ===================================================================
// 13. Initializer List Constructor
// ===================================================================

TEST(UnorderedMapTest, InitializerListConstructor) {
    PrintSection("Initializer List Constructor",
                 "Brace-initializer syntax provides concise map construction.");

    UnorderedMap<std::string, int> map({{"a", 1}, {"b", 2}, {"c", 3}});
    EXPECT_EQ(map.Size(), 3u);
    EXPECT_EQ(map.At("a"), 1);
    EXPECT_EQ(map.At("c"), 3);
}

// ===================================================================
// 14. Bucket Info
// ===================================================================

TEST(UnorderedMapTest, BucketInfo) {
    PrintSection("Bucket Info",
                 "BucketCount, BucketSize, and Bucket provide introspection "
                 "into the internal hash table layout.");

    UnorderedMap<int, int> map(8);
    map.Insert(1, 10);
    EXPECT_GE(map.BucketCount(), 8u);
    EXPECT_NO_THROW((void)map.BucketSize(0));
    EXPECT_THROW((void)map.BucketSize(999999), std::out_of_range);

    std::size_t bucket_idx = map.Bucket(1);
    EXPECT_LT(bucket_idx, map.BucketCount());
}

// ===================================================================
// 15. Large-Scale Stress Test
// ===================================================================

TEST(UnorderedMapTest, LargeScaleStress) {
    PrintSection("Large-Scale Stress Test",
                 "Inserting and looking up 100k elements validates that "
                 "rehashing preserves all data under heavy load.");

    constexpr int kN = 100000;
    UnorderedMap<int, int> map(4, 1.0);  // start small, force many rehashes

    for (int i = 0; i < kN; ++i) {
        map.Insert(i, i);
    }

    EXPECT_EQ(map.Size(), static_cast<std::size_t>(kN));

    for (int i = 0; i < kN; ++i) {
        ASSERT_TRUE(map.Contains(i)) << "Missing key " << i;
        auto* val = map.Find(i);
        ASSERT_NE(val, nullptr);
        EXPECT_EQ(*val, i);
    }

    std::cout << "  Final bucket count: " << map.BucketCount() << "\n";
    std::cout << "  Final load factor:  " << map.LoadFactor() << "\n";
}

// ===================================================================
// 16. Manual Rehash
// ===================================================================

TEST(UnorderedMapTest, ManualRehash) {
    PrintSection("Manual Rehash",
                 "Rehash(n) lets callers pre-size the bucket array, "
                 "avoiding incremental rehashing during bulk inserts.");

    UnorderedMap<int, int> map;
    map.Insert(1, 10);
    map.Insert(2, 20);

    map.Rehash(64);
    EXPECT_GE(map.BucketCount(), 64u);

    // Data survives
    EXPECT_EQ(map.Size(), 2u);
    EXPECT_EQ(map.At(1), 10);
    EXPECT_EQ(map.At(2), 20);
}
