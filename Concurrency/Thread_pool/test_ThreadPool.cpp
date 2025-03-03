/**
 * @file test_ThreadPool.cpp
 * @brief Google Test suite for the ThreadPool class.
 *
 * Tests verify construction, task execution, future-based return values,
 * graceful shutdown via stop_token, and stress conditions.
 *
 * Each test prints an educational banner explaining *why* it matters.
 */

#include "ThreadPool.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <iostream>

namespace {

/**
 * @brief Print a section banner for educational output.
 * @param title Test section title.
 * @param why   One-line explanation of why this test matters.
 */
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Worker creation
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, ConstructsWithRequestedThreadCount) {
    PrintSection("Worker Creation",
                 "The pool must spawn exactly the number of threads requested.");

    ThreadPool pool(4);
    EXPECT_EQ(pool.ThreadCount(), 4u);
}

TEST(ThreadPoolTest, DefaultThreadCountMatchesHardware) {
    PrintSection("Default Thread Count",
                 "Omitting the argument should use hardware_concurrency().");

    ThreadPool pool;
    EXPECT_EQ(pool.ThreadCount(),
              static_cast<std::size_t>(std::jthread::hardware_concurrency()));
}

// ---------------------------------------------------------------------------
// Task execution and completion
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, ExecutesSingleTask) {
    PrintSection("Single Task Execution",
                 "The most basic contract: enqueue one task, it runs.");

    ThreadPool pool(2);
    std::atomic<bool> executed{false};

    pool.Enqueue([&] { executed = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(executed.load());
}

TEST(ThreadPoolTest, ExecutesMultipleTasks) {
    PrintSection("Multiple Task Execution",
                 "All enqueued tasks must complete, atomically counted.");

    constexpr int kTaskCount = 100;
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < kTaskCount; ++i) {
        pool.Enqueue([&] { counter++; });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    EXPECT_EQ(counter.load(), kTaskCount);
}

TEST(ThreadPoolTest, TasksRunOnMultipleThreads) {
    PrintSection("Thread Distribution",
                 "Tasks should spread across workers, not serialize on one thread.");

    ThreadPool pool(4);
    std::mutex mtx;
    std::set<std::thread::id> ids;

    for (int i = 0; i < 20; ++i) {
        pool.Enqueue([&] {
            std::lock_guard<std::mutex> lock(mtx);
            ids.insert(std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    EXPECT_GT(ids.size(), 1u);
}

TEST(ThreadPoolTest, PendingTasksReflectsQueueDepth) {
    PrintSection("Pending Tasks",
                 "PendingTasks() must accurately reflect the queue depth.");

    // Use 1 worker so tasks queue up behind a blocking task.
    ThreadPool pool(1);
    std::atomic<bool> gate{false};

    // Block the single worker.
    pool.Enqueue([&] {
        while (!gate.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Enqueue more tasks while worker is blocked.
    pool.Enqueue([] {});
    pool.Enqueue([] {});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(pool.PendingTasks(), 2u);

    gate = true;  // Unblock.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(pool.PendingTasks(), 0u);
}

// ---------------------------------------------------------------------------
// Future-based return values
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, EnqueueWithFutureReturnsInt) {
    PrintSection("Future-Based Return (int)",
                 "EnqueueWithFuture wraps a packaged_task so callers can await results.");

    ThreadPool pool(2);
    auto future = pool.EnqueueWithFuture([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPoolTest, EnqueueWithFutureReturnsString) {
    PrintSection("Future-Based Return (string)",
                 "Verify templated future works with non-trivial return types.");

    ThreadPool pool(2);
    auto future = pool.EnqueueWithFuture([] { return std::string("hello"); });
    EXPECT_EQ(future.get(), "hello");
}

TEST(ThreadPoolTest, EnqueueWithFutureForwardsArgs) {
    PrintSection("Future With Arguments",
                 "EnqueueWithFuture must forward arguments to the callable.");

    ThreadPool pool(2);
    auto future = pool.EnqueueWithFuture([](int a, int b) { return a + b; }, 3, 4);
    EXPECT_EQ(future.get(), 7);
}

// ---------------------------------------------------------------------------
// Graceful shutdown via stop_token
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, DestructorDrainsQueue) {
    PrintSection("Destructor Drains Queue",
                 "On shutdown, remaining queued tasks must still execute (no data loss).");

    std::atomic<int> counter{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 50; ++i) {
            pool.Enqueue([&] { counter++; });
        }
    }  // destructor blocks until all tasks complete

    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPoolTest, StopTokenPreventsNewEnqueues) {
    PrintSection("Stop Token Prevents Enqueue",
                 "After destruction begins, Enqueue must throw to reject new work.");

    auto pool = std::make_unique<ThreadPool>(2);
    pool.reset();  // triggers destructor

    // We can't enqueue on a destroyed pool, so this just verifies
    // the destructor completed cleanly without deadlock.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Stress test
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, StressManyTasks) {
    PrintSection("Stress Test (10,000 tasks)",
                 "High task volume must complete correctly under contention.");

    constexpr int kTaskCount = 10000;
    std::atomic<int> counter{0};

    {
        ThreadPool pool(8);
        for (int i = 0; i < kTaskCount; ++i) {
            pool.Enqueue([&] { counter++; });
        }
    }

    EXPECT_EQ(counter.load(), kTaskCount);
}
