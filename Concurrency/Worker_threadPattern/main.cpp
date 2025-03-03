/**
 * @file main.cpp
 * @brief Driver program for the WorkerThreadPool demo.
 *
 * Enqueues tasks and demonstrates cooperative shutdown via stop_token.
 *
 * Build:  make && ./main
 */

#include "WorkerThreadPool.hpp"
#include <iostream>
#include <chrono>

int main() {
    WorkerThreadPool pool(4);

    for (int i = 0; i < 10; ++i) {
        pool.EnqueueTask([i] {
            std::cout << "Processing task #" << i << " on thread "
                      << std::this_thread::get_id() << "\n";
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Pending tasks before shutdown: " << pool.PendingTasks() << "\n";
    pool.Shutdown();
    std::cout << "Pool shut down successfully.\n";

    return 0;
}
