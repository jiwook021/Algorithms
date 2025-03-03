/**
 * @file main.cpp
 * @brief Interactive driver for the ThreadPool.
 *
 * Shows why thread pools exist, demonstrates fire-and-forget tasks,
 * future-based parallel computation, and sequential-vs-parallel
 * speedup comparison.
 */

#include "ThreadPool.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Mutex for clean multi-threaded console output
static std::mutex g_cout_mutex;

static void Print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cout << msg << std::flush;
}

/**
 * @brief Simulate CPU work: compute sum of sqrt(1) + sqrt(2) + ... + sqrt(n).
 */
static long long HeavyCompute(int n) {
    long long sum = 0;
    for (int i = 1; i <= n; ++i) {
        sum += static_cast<long long>(std::sqrt(static_cast<double>(i)));
    }
    return sum;
}

int main() {
    unsigned int hw = std::jthread::hardware_concurrency();

    std::cout << "=============================================\n"
              << "  Thread Pool\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  A thread pool pre-creates a fixed number of worker\n"
              << "  threads that wait for tasks.  When work arrives, an\n"
              << "  idle worker picks it up immediately -- no thread\n"
              << "  creation overhead.\n\n"

              << "WHY IT'S NEEDED\n"
              << "  Creating a thread is expensive (~0.1-1 ms on Linux).\n"
              << "  If your server handles 1000 requests/sec, spawning a\n"
              << "  new thread for each wastes up to 1 second just on\n"
              << "  thread creation.  A pool creates threads once and\n"
              << "  reuses them, turning that overhead into nearly zero.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Constructor spawns N worker threads (jthread).\n"
              << "  2. Workers wait on a condition variable for tasks.\n"
              << "  3. Enqueue() pushes a task and wakes one worker.\n"
              << "  4. Worker pops the task, executes it, waits again.\n"
              << "  5. EnqueueWithFuture() wraps the task in a\n"
              << "     packaged_task so the caller can get the result.\n"
              << "  6. Destructor signals stop via stop_token, drains\n"
              << "     remaining tasks, and joins all workers.\n\n"

              << "C++20 FEATURES USED\n"
              << "  jthread     : auto-joins on destruction (no manual join).\n"
              << "  stop_token  : cooperative cancellation (no bool flag).\n"
              << "  cv_any::wait: returns immediately when stop requested.\n\n"

              << "  This machine has " << hw << " hardware threads.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Demo 1: Fire-and-forget tasks (web server simulation)
    // -----------------------------------------------------------------

    std::cout << "DEMO 1: Web Server Simulation (Fire-and-Forget)\n\n"
              << "  A web server with 4 worker threads handles 12\n"
              << "  incoming requests.  Each request takes ~100 ms.\n"
              << "  With 4 workers, ~4 requests run in parallel.\n\n";

    {
        ThreadPool pool(4);
        std::mutex id_mutex;
        std::map<std::thread::id, int> thread_ids;
        int next_id = 1;

        for (int i = 1; i <= 12; ++i) {
            pool.Enqueue([i, &id_mutex, &thread_ids, &next_id] {
                int worker;
                {
                    std::lock_guard<std::mutex> lock(id_mutex);
                    auto tid = std::this_thread::get_id();
                    if (thread_ids.find(tid) == thread_ids.end()) {
                        thread_ids[tid] = next_id++;
                    }
                    worker = thread_ids[tid];
                }

                std::ostringstream oss;
                oss << "    Request " << std::setw(2) << i
                    << "  -> Worker " << worker
                    << "  (processing...)\n";
                Print(oss.str());

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            });
        }
    } // pool destructor waits for all tasks to finish

    std::cout << "\n  All 12 requests handled.  4 workers shared the load.\n"
              << "  Without a pool, we'd create and destroy 12 threads.\n";

    // -----------------------------------------------------------------
    // Demo 2: Future-based tasks (parallel computation)
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  DEMO 2: Parallel Computation (Futures)\n"
              << "---------------------------------------------\n\n"
              << "  EnqueueWithFuture() lets you submit a task and get\n"
              << "  a std::future back.  Calling future.get() blocks\n"
              << "  until the result is ready.\n\n"
              << "  Computing sum of sqrt(1..N) for several values of N:\n\n";

    {
        ThreadPool pool(4);

        struct Job {
            int n;
            const char* label;
        };
        std::vector<Job> jobs = {
            {1000000,  " 1M"},
            {5000000,  " 5M"},
            {10000000, "10M"},
            {20000000, "20M"},
        };

        std::vector<std::future<long long>> futures;
        for (const auto& job : jobs) {
            futures.push_back(pool.EnqueueWithFuture(HeavyCompute, job.n));
        }

        std::cout << "    Submitted " << jobs.size()
                  << " tasks, now collecting results...\n\n";

        for (std::size_t i = 0; i < jobs.size(); ++i) {
            long long result = futures[i].get();
            std::cout << "    sqrt_sum(" << jobs[i].label << ") = "
                      << std::setw(12) << result << "\n";
        }
    }

    std::cout << "\n  All 4 computations ran in parallel across workers.\n";

    // -----------------------------------------------------------------
    // Demo 3: Sequential vs Parallel speedup
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  DEMO 3: Sequential vs Parallel Speedup\n"
              << "---------------------------------------------\n\n"
              << "  Thread pools shine with I/O-bound work: network calls,\n"
              << "  file reads, database queries.  While one thread waits\n"
              << "  for I/O, others keep working.\n\n"
              << "  Simulating 8 \"API calls\" that each take 200 ms:\n\n";

    constexpr int kWorkItems = 8;
    constexpr auto kTaskDuration = std::chrono::milliseconds(200);

    auto simulate_api_call = [&](int id) {
        std::this_thread::sleep_for(kTaskDuration);
        return id * 10;
    };

    // Sequential
    auto seq_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kWorkItems; ++i) {
        simulate_api_call(i);
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        seq_end - seq_start).count();

    // Parallel with 4 threads
    auto par_start = std::chrono::high_resolution_clock::now();
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < kWorkItems; ++i) {
            futures.push_back(
                pool.EnqueueWithFuture(simulate_api_call, i));
        }
        for (auto& f : futures) {
            f.get();
        }
    }
    auto par_end = std::chrono::high_resolution_clock::now();
    auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        par_end - par_start).count();

    double speedup = (par_ms > 0)
        ? static_cast<double>(seq_ms) / static_cast<double>(par_ms)
        : 0.0;

    std::cout << "    Sequential (1 thread):   " << std::setw(5)
              << seq_ms << " ms  (8 x 200 ms = 1600 ms)\n"
              << "    Parallel   (4 threads):  " << std::setw(5)
              << par_ms << " ms  (2 batches x 200 ms = 400 ms)\n"
              << "    Speedup:                 " << std::fixed
              << std::setprecision(1) << speedup << "x\n\n"
              << "  While thread A waits for its \"API response\", threads\n"
              << "  B, C, D are making their own calls.  This is why web\n"
              << "  servers, database clients, and downloaders use pools.\n";

    // -----------------------------------------------------------------
    // Demo 4: Graceful shutdown
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  DEMO 4: Graceful Shutdown\n"
              << "---------------------------------------------\n\n"
              << "  When the pool is destroyed:\n"
              << "    1. jthread's destructor calls request_stop().\n"
              << "    2. Workers finish their current task.\n"
              << "    3. Workers drain any remaining queued tasks.\n"
              << "    4. jthread's destructor calls join() automatically.\n"
              << "  No tasks are lost -- every enqueued task completes.\n\n";

    {
        std::atomic<int> completed{0};
        {
            ThreadPool pool(2);
            for (int i = 0; i < 20; ++i) {
                pool.Enqueue([&completed] {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(10));
                    completed++;
                });
            }
            std::cout << "    Enqueued 20 tasks, destroying pool now...\n";
        } // destructor blocks until all 20 finish
        std::cout << "    Pool destroyed.  Tasks completed: "
                  << completed.load() << "/20\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
