/**
 * @file test_WorkerThreadPool.cpp
 * @brief Google Test suite for WorkerThreadPool (jthread + stop_token).
 *
 * Tests verify construction, cooperative shutdown via stop_token, task
 * execution, exception safety, pending-task draining, and concurrency stress.
 */

#include "WorkerThreadPool.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, ConstructsWithGivenThreadCount) {
    PrintSection("ConstructsWithGivenThreadCount",
                 "Verify the pool spawns exactly the requested number of workers.");
    WorkerThreadPool pool(4);
    EXPECT_EQ(pool.ThreadCount(), 4u);
    EXPECT_TRUE(pool.IsRunning());
}

TEST(WorkerThreadPoolTest, ConstructsWithSingleThread) {
    PrintSection("ConstructsWithSingleThread",
                 "Edge case: pool with one worker must still function.");
    WorkerThreadPool pool(1);
    EXPECT_EQ(pool.ThreadCount(), 1u);
    EXPECT_TRUE(pool.IsRunning());
}

// ---------------------------------------------------------------------------
// Task execution
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, ExecutesSingleTask) {
    PrintSection("ExecutesSingleTask",
                 "Ensure the basic enqueue-and-run path works.");
    WorkerThreadPool pool(2);
    std::atomic<bool> executed{false};
    pool.EnqueueTask([&] { executed = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.Shutdown();
    EXPECT_TRUE(executed.load());
}

TEST(WorkerThreadPoolTest, ExecutesMultipleTasks) {
    PrintSection("ExecutesMultipleTasks",
                 "All enqueued tasks must complete before shutdown returns.");
    WorkerThreadPool pool(4);
    constexpr int kNumTasks = 100;
    std::atomic<int> counter{0};
    for (int i = 0; i < kNumTasks; ++i) {
        pool.EnqueueTask([&] { counter++; });
    }
    pool.Shutdown();
    EXPECT_EQ(counter.load(), kNumTasks);
}

TEST(WorkerThreadPoolTest, TasksRunOnDifferentThreads) {
    PrintSection("TasksRunOnDifferentThreads",
                 "Work should distribute across multiple worker threads.");
    WorkerThreadPool pool(4);
    std::mutex mtx;
    std::set<std::thread::id> thread_ids;
    for (int i = 0; i < 20; ++i) {
        pool.EnqueueTask([&] {
            std::lock_guard<std::mutex> lock(mtx);
            thread_ids.insert(std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    pool.Shutdown();
    EXPECT_GT(thread_ids.size(), 1u);
}

// ---------------------------------------------------------------------------
// Cooperative shutdown via stop_token
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, ShutdownRequestsStopOnAllWorkers) {
    PrintSection("ShutdownRequestsStopOnAllWorkers",
                 "Shutdown must use jthread::request_stop -- cooperative cancellation.");
    WorkerThreadPool pool(4);
    EXPECT_TRUE(pool.IsRunning());
    pool.Shutdown();
    EXPECT_FALSE(pool.IsRunning());
}

TEST(WorkerThreadPoolTest, ShutdownIsIdempotent) {
    PrintSection("ShutdownIsIdempotent",
                 "Calling Shutdown() twice must not crash or deadlock.");
    WorkerThreadPool pool(2);
    pool.Shutdown();
    pool.Shutdown();  // second call -- must be harmless
    EXPECT_FALSE(pool.IsRunning());
}

TEST(WorkerThreadPoolTest, DestructorCallsShutdown) {
    PrintSection("DestructorCallsShutdown",
                 "Destroying the pool without explicit Shutdown must still join workers.");
    std::atomic<int> counter{0};
    {
        WorkerThreadPool pool(2);
        for (int i = 0; i < 10; ++i) {
            pool.EnqueueTask([&] { counter++; });
        }
        // destructor fires here
    }
    EXPECT_EQ(counter.load(), 10);
}

// ---------------------------------------------------------------------------
// Pending tasks drain on shutdown
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, ShutdownDrainsQueuedTasks) {
    PrintSection("ShutdownDrainsQueuedTasks",
                 "Shutdown must execute all already-enqueued tasks before returning.");
    WorkerThreadPool pool(1);
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; ++i) {
        pool.EnqueueTask([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter++;
        });
    }
    pool.Shutdown();
    EXPECT_EQ(counter.load(), 10);
}

TEST(WorkerThreadPoolTest, PendingTasksDecreasesAfterProcessing) {
    PrintSection("PendingTasksDecreasesAfterProcessing",
                 "PendingTasks() should reach 0 after all work is consumed.");
    WorkerThreadPool pool(1);
    std::atomic<bool> hold{true};

    // Block the single worker so queued tasks pile up.
    pool.EnqueueTask([&] {
        while (hold.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    pool.EnqueueTask([] {});
    pool.EnqueueTask([] {});
    EXPECT_GE(pool.PendingTasks(), 1u);

    hold = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();
    EXPECT_EQ(pool.PendingTasks(), 0u);
}

// ---------------------------------------------------------------------------
// Exception safety
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, ThrowingTaskDoesNotCrashWorker) {
    PrintSection("ThrowingTaskDoesNotCrashWorker",
                 "A task that throws must not terminate the worker thread.");
    WorkerThreadPool pool(2);
    std::atomic<int> counter{0};

    pool.EnqueueTask([] { throw std::runtime_error("intentional"); });
    pool.EnqueueTask([&] { counter++; });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();
    EXPECT_EQ(counter.load(), 1);
}

TEST(WorkerThreadPoolTest, MultipleThrowingTasksDoNotKillPool) {
    PrintSection("MultipleThrowingTasksDoNotKillPool",
                 "Several consecutive throwing tasks must not crash the pool.");
    WorkerThreadPool pool(2);
    std::atomic<int> good_count{0};

    for (int i = 0; i < 5; ++i) {
        pool.EnqueueTask([] { throw std::runtime_error("boom"); });
    }
    for (int i = 0; i < 5; ++i) {
        pool.EnqueueTask([&] { good_count++; });
    }

    pool.Shutdown();
    EXPECT_EQ(good_count.load(), 5);
}

// ---------------------------------------------------------------------------
// Stress
// ---------------------------------------------------------------------------

TEST(WorkerThreadPoolTest, StressTestManyConcurrentTasks) {
    PrintSection("StressTestManyConcurrentTasks",
                 "High-volume enqueue must not lose tasks or deadlock.");
    WorkerThreadPool pool(8);
    constexpr int kNumTasks = 10000;
    std::atomic<int> counter{0};
    for (int i = 0; i < kNumTasks; ++i) {
        pool.EnqueueTask([&] { counter++; });
    }
    pool.Shutdown();
    EXPECT_EQ(counter.load(), kNumTasks);
}
