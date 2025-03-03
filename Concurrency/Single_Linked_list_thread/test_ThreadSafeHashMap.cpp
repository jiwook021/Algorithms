/**
 * @file test_ThreadSafeHashMap.cpp
 * @brief Unit tests for the ThreadSafeHashMap class.
 *
 * Tests verify insert/find, update, erase, size tracking, clear,
 * load factor, invalid construction, and concurrent access safety.
 */

#include "ThreadSafeHashMap.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <string>

/** @brief Verify insert and find operations. */
static void TestInsertAndFind() {
    ThreadSafeHashMap<int, std::string> map;
    map.Insert(1, "one");
    map.Insert(2, "two");

    auto v1 = map.Find(1);
    auto v2 = map.Find(2);
    auto v3 = map.Find(3);

    assert(v1.has_value() && *v1 == "one");
    assert(v2.has_value() && *v2 == "two");
    assert(!v3.has_value());
    std::cout << "[PASS] TestInsertAndFind\n";
}

/** @brief Verify updating an existing key. */
static void TestUpdateExistingKey() {
    ThreadSafeHashMap<int, std::string> map;
    map.Insert(1, "old");
    map.Insert(1, "new");
    auto v = map.Find(1);
    assert(v.has_value() && *v == "new");
    std::cout << "[PASS] TestUpdateExistingKey\n";
}

/** @brief Verify erase operation. */
static void TestEraseKey() {
    ThreadSafeHashMap<int, std::string> map;
    map.Insert(1, "one");
    assert(map.Erase(1));
    assert(!map.Erase(1));
    assert(!map.Find(1).has_value());
    std::cout << "[PASS] TestEraseKey\n";
}

/** @brief Verify size and empty tracking. */
static void TestSizeAndEmpty() {
    ThreadSafeHashMap<int, int> map;
    assert(map.Empty());
    assert(map.Size() == 0u);
    map.Insert(1, 10);
    map.Insert(2, 20);
    assert(map.Size() == 2u);
    assert(!map.Empty());
    std::cout << "[PASS] TestSizeAndEmpty\n";
}

/** @brief Verify clear removes all elements. */
static void TestClear() {
    ThreadSafeHashMap<int, int> map;
    for (int i = 0; i < 10; ++i) map.Insert(i, i * 10);
    assert(map.Size() == 10u);
    map.Clear();
    assert(map.Size() == 0u);
    assert(map.Empty());
    std::cout << "[PASS] TestClear\n";
}

/** @brief Verify load factor computation. */
static void TestLoadFactor() {
    ThreadSafeHashMap<int, int> map(16, 0.75f);
    assert(map.LoadFactor() == 0.0f);
    for (int i = 0; i < 8; ++i) map.Insert(i, i);
    assert(map.LoadFactor() > 0.0f);
    std::cout << "[PASS] TestLoadFactor\n";
}

/** @brief Verify invalid constructor arguments throw. */
static void TestInvalidConstructorArgs() {
    bool caught = false;
    try { ThreadSafeHashMap<int, int> m(0, 0.5f); }
    catch (const std::invalid_argument&) { caught = true; }
    assert(caught);

    caught = false;
    try { ThreadSafeHashMap<int, int> m(16, 0.0f); }
    catch (const std::invalid_argument&) { caught = true; }
    assert(caught);

    caught = false;
    try { ThreadSafeHashMap<int, int> m(16, 1.0f); }
    catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
    std::cout << "[PASS] TestInvalidConstructorArgs\n";
}

/** @brief Verify concurrent insert and find do not crash or lose data. */
static void TestConcurrentInsertAndFind() {
    ThreadSafeHashMap<int, int> map(1024, 0.75f);

    auto writer = [&](int start) {
        for (int i = start; i < start + 100; ++i)
            map.Insert(i, i * 10);
    };
    auto reader = [&](int start) {
        for (int i = start; i < start + 100; ++i)
            map.Find(i);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back(writer, t * 100);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back(reader, t * 100);
    for (auto& t : threads) t.join();

    assert(map.Size() == 400u);
    std::cout << "[PASS] TestConcurrentInsertAndFind\n";
}

/** @brief Verify concurrent mixed operations do not crash. */
static void TestConcurrentMixedOperations() {
    ThreadSafeHashMap<int, int> map(1024, 0.75f);
    for (int i = 0; i < 100; ++i) map.Insert(i, i * 10);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&map, t]() {
            for (int i = 0; i < 500; ++i) {
                int key = i % 100;
                if ((i + t) % 3 == 0)      map.Insert(key, t * 1000 + i);
                else if ((i + t) % 3 == 1)  map.Find(key);
                else                         map.Erase(key);
            }
        });
    }
    for (auto& t : threads) t.join();
    // Should not crash; size is non-deterministic
    std::cout << "[PASS] TestConcurrentMixedOperations\n";
}

int main() {
    TestInsertAndFind();
    TestUpdateExistingKey();
    TestEraseKey();
    TestSizeAndEmpty();
    TestClear();
    TestLoadFactor();
    TestInvalidConstructorArgs();
    TestConcurrentInsertAndFind();
    TestConcurrentMixedOperations();

    std::cout << "\nAll ThreadSafeHashMap tests passed.\n";
    return 0;
}
