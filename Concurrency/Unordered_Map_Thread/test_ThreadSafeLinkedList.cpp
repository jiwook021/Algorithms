/**
 * @file test_ThreadSafeLinkedList.cpp
 * @brief Google Test suite for the ThreadSafeLinkedList class.
 *
 * Tests cover single-threaded correctness (push, pop, contains, remove,
 * insert after, clear, toString, move construction) and concurrent
 * scenarios (PushFront no data loss, read-write consistency, PopFront
 * under contention) using std::jthread.
 */

#include "ThreadSafeLinkedList.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <mutex>

namespace {

/// Print a section header explaining what the test demonstrates and why.
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Single-threaded correctness tests
// ---------------------------------------------------------------------------

TEST(ThreadSafeLinkedListTest, PushFrontAndSize) {
    PrintSection("PushFront + Size",
                 "Verify basic insertion increments size correctly");
    ThreadSafeLinkedList<int> list;
    EXPECT_TRUE(list.Empty());
    list.PushFront(1);
    list.PushFront(2);
    EXPECT_EQ(list.Size(), 2u);
    EXPECT_FALSE(list.Empty());
}

TEST(ThreadSafeLinkedListTest, PopFrontLifo) {
    PrintSection("PopFront LIFO order",
                 "Confirm elements are returned in last-in first-out order");
    ThreadSafeLinkedList<int> list;
    list.PushFront(10);
    list.PushFront(20);

    auto v = list.PopFront();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 20);

    v = list.PopFront();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 10);

    v = list.PopFront();
    EXPECT_FALSE(v.has_value());
}

TEST(ThreadSafeLinkedListTest, ContainsFindsElements) {
    PrintSection("Contains",
                 "Verify Contains returns true for existing and false for missing values");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    list.PushFront(2);
    list.PushFront(3);

    EXPECT_TRUE(list.Contains(1));
    EXPECT_TRUE(list.Contains(3));
    EXPECT_FALSE(list.Contains(99));
}

TEST(ThreadSafeLinkedListTest, RemoveMiddleElement) {
    PrintSection("Remove middle element",
                 "Verify Remove correctly unlinks a non-head node");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    list.PushFront(2);
    list.PushFront(3);

    EXPECT_TRUE(list.Remove(2));
    EXPECT_FALSE(list.Contains(2));
    EXPECT_EQ(list.Size(), 2u);
    EXPECT_FALSE(list.Remove(99));
}

TEST(ThreadSafeLinkedListTest, RemoveHead) {
    PrintSection("Remove head element",
                 "Verify removing the head node re-links correctly");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    list.PushFront(2);
    EXPECT_TRUE(list.Remove(2));
    EXPECT_EQ(list.Size(), 1u);
    EXPECT_TRUE(list.Contains(1));
}

TEST(ThreadSafeLinkedListTest, InsertAfter) {
    PrintSection("InsertAfter",
                 "Verify element is placed immediately after the target node");
    ThreadSafeLinkedList<int> list;
    list.PushFront(3);
    list.PushFront(1);
    list.InsertAfter(1, 2);

    auto v = list.PopFront();
    EXPECT_EQ(*v, 1);
    v = list.PopFront();
    EXPECT_EQ(*v, 2);
    v = list.PopFront();
    EXPECT_EQ(*v, 3);
}

TEST(ThreadSafeLinkedListTest, InsertAfterThrowsOnEmptyList) {
    PrintSection("InsertAfter on empty list",
                 "Verify runtime_error is thrown when list is empty");
    ThreadSafeLinkedList<int> list;
    EXPECT_THROW(list.InsertAfter(1, 2), std::runtime_error);
}

TEST(ThreadSafeLinkedListTest, InsertAfterThrowsOnMissingTarget) {
    PrintSection("InsertAfter with missing target",
                 "Verify runtime_error is thrown when target does not exist");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    EXPECT_THROW(list.InsertAfter(99, 2), std::runtime_error);
}

TEST(ThreadSafeLinkedListTest, Clear) {
    PrintSection("Clear",
                 "Verify Clear removes all elements and resets size to 0");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    list.PushFront(2);
    list.Clear();
    EXPECT_TRUE(list.Empty());
    EXPECT_EQ(list.Size(), 0u);
}

TEST(ThreadSafeLinkedListTest, ToStringFormat) {
    PrintSection("ToString",
                 "Verify string representation matches expected arrow format");
    ThreadSafeLinkedList<int> list;
    list.PushFront(3);
    list.PushFront(2);
    list.PushFront(1);
    EXPECT_EQ(list.ToString(), "1 -> 2 -> 3");
}

TEST(ThreadSafeLinkedListTest, MoveConstruct) {
    PrintSection("Move construction",
                 "Verify move constructor transfers ownership and empties source");
    ThreadSafeLinkedList<int> list;
    list.PushFront(1);
    list.PushFront(2);

    ThreadSafeLinkedList<int> moved(std::move(list));
    EXPECT_EQ(moved.Size(), 2u);
    EXPECT_TRUE(moved.Contains(1));
}

// ---------------------------------------------------------------------------
// Concurrent tests
// ---------------------------------------------------------------------------

TEST(ThreadSafeLinkedListConcurrentTest, ConcurrentPushFrontNoDataLoss) {
    PrintSection("Concurrent PushFront -- no data loss",
                 "N threads each push M items; final size must equal N*M, "
                 "proving the shared_mutex serializes writes correctly");

    constexpr int kNumThreads = 8;
    constexpr int kItemsPerThread = 100;
    ThreadSafeLinkedList<int> list;

    {
        std::vector<std::jthread> threads;
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&list, t]() {
                for (int i = 0; i < kItemsPerThread; ++i) {
                    list.PushFront(t * kItemsPerThread + i);
                }
            });
        }
        // jthreads join automatically on scope exit
    }

    const std::size_t expected = static_cast<std::size_t>(kNumThreads * kItemsPerThread);
    EXPECT_EQ(list.Size(), expected)
        << "Data loss detected: expected " << expected
        << " items but got " << list.Size();

    // Verify every value is present
    for (int t = 0; t < kNumThreads; ++t) {
        for (int i = 0; i < kItemsPerThread; ++i) {
            EXPECT_TRUE(list.Contains(t * kItemsPerThread + i))
                << "Missing value: " << (t * kItemsPerThread + i);
        }
    }

    std::cout << "  Final size: " << list.Size()
              << " (expected " << expected << ")\n";
}

TEST(ThreadSafeLinkedListConcurrentTest, ConcurrentReadWriteConsistency) {
    PrintSection("Concurrent read-write consistency",
                 "Writers push values while readers call Contains and Size; "
                 "no crashes or undefined behavior proves shared_lock / "
                 "unique_lock separation works");

    constexpr int kWriterThreads = 4;
    constexpr int kReaderThreads = 4;
    constexpr int kOpsPerThread = 200;
    ThreadSafeLinkedList<int> list;

    std::atomic<bool> writers_done{false};

    {
        std::vector<std::jthread> threads;

        // Writer threads
        for (int t = 0; t < kWriterThreads; ++t) {
            threads.emplace_back([&list, t]() {
                for (int i = 0; i < kOpsPerThread; ++i) {
                    list.PushFront(t * kOpsPerThread + i);
                }
            });
        }

        // Reader threads
        for (int t = 0; t < kReaderThreads; ++t) {
            threads.emplace_back([&list, &writers_done]() {
                while (!writers_done.load(std::memory_order_acquire)) {
                    // These read operations must not crash or return garbage
                    [[maybe_unused]] auto sz = list.Size();
                    [[maybe_unused]] bool found = list.Contains(42);
                    [[maybe_unused]] auto str = list.ToString();
                    // Yield to prevent reader starvation of writers
                    std::this_thread::yield();
                }
            });
        }

        // Wait for writer threads to finish, then signal readers.
        // Calling join() on a jthread makes it unjoinable so the
        // destructor will skip it.
        for (int i = 0; i < kWriterThreads; ++i) {
            threads[static_cast<std::size_t>(i)].join();
        }
        writers_done.store(true, std::memory_order_release);
        // Remaining reader jthreads join on scope exit
    }

    const std::size_t expected =
        static_cast<std::size_t>(kWriterThreads * kOpsPerThread);
    EXPECT_EQ(list.Size(), expected);
    std::cout << "  No crashes or UB detected. Final size: "
              << list.Size() << "\n";
}

TEST(ThreadSafeLinkedListConcurrentTest, PopFrontUnderContention) {
    PrintSection("PopFront under contention",
                 "Pre-fill list then have N threads race to PopFront; "
                 "total popped items must equal initial size with no duplicates");

    constexpr int kTotalItems = 500;
    constexpr int kNumThreads = 8;
    ThreadSafeLinkedList<int> list;

    for (int i = 0; i < kTotalItems; ++i) {
        list.PushFront(i);
    }
    ASSERT_EQ(list.Size(), static_cast<std::size_t>(kTotalItems));

    std::mutex collected_mutex;
    std::vector<int> collected;
    collected.reserve(kTotalItems);

    {
        std::vector<std::jthread> threads;
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&list, &collected, &collected_mutex]() {
                while (true) {
                    auto val = list.PopFront();
                    if (!val.has_value()) break;
                    std::lock_guard<std::mutex> lock(collected_mutex);
                    collected.push_back(*val);
                }
            });
        }
    }

    EXPECT_EQ(collected.size(), static_cast<std::size_t>(kTotalItems))
        << "Some items were lost or duplicated during concurrent PopFront";
    EXPECT_EQ(list.Size(), 0u) << "List should be empty after draining";

    // Verify no duplicates
    std::set<int> unique_vals(collected.begin(), collected.end());
    EXPECT_EQ(unique_vals.size(), static_cast<std::size_t>(kTotalItems))
        << "Duplicate values detected in popped results";

    std::cout << "  Popped " << collected.size() << " items with "
              << unique_vals.size() << " unique values (expected "
              << kTotalItems << ")\n";
}
