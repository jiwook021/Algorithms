/**
 * @file main.cpp
 * @brief Driver program for the DiningPhilosophers simulation.
 *
 * Spawns N philosopher threads using std::jthread that each think-eat-think
 * for a fixed number of rounds, demonstrating deadlock-free dining via the
 * monitor pattern.
 *
 * Build:  make && ./main
 */

#include "DiningPhilosophers.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>

int main() {
    constexpr std::size_t kNumPhilosophers = 5;
    constexpr int kNumRounds = 3;

    DiningPhilosophers table(kNumPhilosophers);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> think_dist(500, 1500);
    std::uniform_int_distribution<> eat_dist(500, 1000);

    auto philosopher_routine = [&](std::size_t id) {
        for (int round = 0; round < kNumRounds; ++round) {
            std::cout << "Philosopher " << id << " thinking\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(think_dist(gen)));

            table.PickupForks(id);
            std::cout << "Philosopher " << id << " eating\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(eat_dist(gen)));

            table.PutdownForks(id);
        }
        std::cout << "Philosopher " << id << " done\n";
    };

    // --- std::jthread: automatically joins on destruction ---
    std::vector<std::jthread> threads;
    threads.reserve(kNumPhilosophers);

    std::cout << "Dining philosophers simulation (" << kNumRounds
              << " rounds each)\n";

    for (std::size_t i = 0; i < kNumPhilosophers; ++i) {
        threads.emplace_back(philosopher_routine, i);
    }

    // jthreads join automatically when the vector destructs, but we join
    // explicitly here so we can print the final message afterwards.
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "All philosophers finished.\n";
    return 0;
}
