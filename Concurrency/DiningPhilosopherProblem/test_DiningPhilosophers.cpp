/**
 * @file test_DiningPhilosophers.cpp
 * @brief Google-Test suite for the DiningPhilosophers monitor.
 *
 * Verifies:
 *   1. No deadlock under contention (all philosophers eat at least once).
 *   2. Mutual exclusion (adjacent philosophers never eat simultaneously).
 *   3. Fairness (all philosophers eat the expected number of meals).
 *
 * Uses std::jthread for philosopher threads throughout.
 */

#include "DiningPhilosophers.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class DiningPhilosophersTest : public ::testing::Test {
protected:
    static constexpr std::size_t kDefaultSize = 5;
};

// ---------------------------------------------------------------------------
// Basic construction & state tests
// ---------------------------------------------------------------------------

TEST_F(DiningPhilosophersTest, ConstructsWithCorrectCount) {
    PrintSection("ConstructsWithCorrectCount",
                 "Verifies constructor stores the philosopher count correctly.");

    DiningPhilosophers dp(kDefaultSize);
    EXPECT_EQ(dp.Count(), kDefaultSize);

    DiningPhilosophers dp3(3);
    EXPECT_EQ(dp3.Count(), 3u);
}

TEST_F(DiningPhilosophersTest, InitialStateIsThinking) {
    PrintSection("InitialStateIsThinking",
                 "All philosophers should start in Thinking state.");

    DiningPhilosophers dp(kDefaultSize);
    for (std::size_t i = 0; i < kDefaultSize; ++i) {
        EXPECT_EQ(dp.GetState(i), DiningPhilosophers::State::Thinking)
            << "Philosopher " << i << " was not Thinking at construction.";
    }
}

TEST_F(DiningPhilosophersTest, SinglePhilosopherCanEat) {
    PrintSection("SinglePhilosopherCanEat",
                 "With no contention a philosopher should transition to Eating "
                 "immediately and back to Thinking after putdown.");

    DiningPhilosophers dp(kDefaultSize);
    dp.PickupForks(0);
    EXPECT_EQ(dp.GetState(0), DiningPhilosophers::State::Eating);
    dp.PutdownForks(0);
    EXPECT_EQ(dp.GetState(0), DiningPhilosophers::State::Thinking);
}

TEST_F(DiningPhilosophersTest, NonAdjacentCanEatSimultaneously) {
    PrintSection("NonAdjacentCanEatSimultaneously",
                 "Philosophers 0 and 2 share no fork and must be able to eat "
                 "at the same time.");

    DiningPhilosophers dp(kDefaultSize);
    dp.PickupForks(0);
    dp.PickupForks(2);
    EXPECT_EQ(dp.GetState(0), DiningPhilosophers::State::Eating);
    EXPECT_EQ(dp.GetState(2), DiningPhilosophers::State::Eating);
    dp.PutdownForks(0);
    dp.PutdownForks(2);
}

TEST_F(DiningPhilosophersTest, StateTransitionCycle) {
    PrintSection("StateTransitionCycle",
                 "Each philosopher's pickup/putdown cycle must return to Thinking.");

    DiningPhilosophers dp(3);
    for (std::size_t i = 0; i < 3; ++i) {
        dp.PickupForks(i);
        EXPECT_EQ(dp.GetState(i), DiningPhilosophers::State::Eating);
        dp.PutdownForks(i);
        EXPECT_EQ(dp.GetState(i), DiningPhilosophers::State::Thinking);
    }
}

// ---------------------------------------------------------------------------
// Concurrency: Mutual exclusion
// ---------------------------------------------------------------------------

TEST_F(DiningPhilosophersTest, MutualExclusion_AdjacentNeverEatSimultaneously) {
    PrintSection("MutualExclusion_AdjacentNeverEatSimultaneously",
                 "The core safety property: no two adjacent philosophers may "
                 "eat at the same instant. CheckNoAdjacentEating() atomically "
                 "snapshots all states under a single lock to avoid TOCTOU races.");

    constexpr std::size_t kPhilosophers = 5;
    constexpr int kRounds = 50;

    DiningPhilosophers dp(kPhilosophers);
    std::atomic<bool> violation{false};
    std::atomic<bool> done{false};
    std::atomic<int> checks{0};

    // Checker thread: continuously samples all states atomically.
    std::jthread checker([&](std::stop_token st) {
        while (!st.stop_requested() && !done.load(std::memory_order_acquire)) {
            if (!dp.CheckNoAdjacentEating()) {
                violation.store(true, std::memory_order_release);
            }
            checks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Philosopher threads (jthread).
    std::vector<std::jthread> philosophers;
    philosophers.reserve(kPhilosophers);
    for (std::size_t id = 0; id < kPhilosophers; ++id) {
        philosophers.emplace_back([&, id] {
            for (int r = 0; r < kRounds; ++r) {
                dp.PickupForks(id);
                // Atomic invariant check while this philosopher is eating.
                if (!dp.CheckNoAdjacentEating()) {
                    violation.store(true, std::memory_order_release);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                dp.PutdownForks(id);
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    for (auto& t : philosophers) { t.join(); }
    done.store(true, std::memory_order_release);
    checker.request_stop();
    checker.join();

    std::cout << "  Invariant checks performed: " << checks.load() << "\n";
    EXPECT_FALSE(violation.load())
        << "Adjacent philosophers were observed eating simultaneously!";
}

// ---------------------------------------------------------------------------
// Concurrency: No deadlock (liveness)
// ---------------------------------------------------------------------------

TEST_F(DiningPhilosophersTest, NoDeadlock_AllPhilosophersComplete) {
    PrintSection("NoDeadlock_AllPhilosophersComplete",
                 "All philosopher threads must finish within a generous timeout. "
                 "If the monitor deadlocks, this test will time out / fail.");

    constexpr std::size_t kPhilosophers = 5;
    constexpr int kRounds = 20;

    DiningPhilosophers dp(kPhilosophers);
    std::atomic<int> total_meals{0};

    auto start = std::chrono::steady_clock::now();

    {
        std::vector<std::jthread> philosophers;
        philosophers.reserve(kPhilosophers);
        for (std::size_t id = 0; id < kPhilosophers; ++id) {
            philosophers.emplace_back([&, id] {
                for (int r = 0; r < kRounds; ++r) {
                    dp.PickupForks(id);
                    total_meals.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    dp.PutdownForks(id);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
        }
        // jthreads auto-join when the vector destructs here.
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    EXPECT_EQ(total_meals.load(), static_cast<int>(kPhilosophers) * kRounds)
        << "Not all meals were completed -- possible deadlock.";

    // Generous upper bound; the real run finishes in tens of milliseconds.
    constexpr long kTimeoutMs = 30000;
    EXPECT_LT(ms, kTimeoutMs)
        << "Simulation took " << ms << " ms -- probable deadlock or starvation.";

    std::cout << "  Completed " << total_meals.load() << " meals in " << ms
              << " ms.\n";
}

// ---------------------------------------------------------------------------
// Concurrency: Fairness
// ---------------------------------------------------------------------------

TEST_F(DiningPhilosophersTest, Fairness_AllPhilosophersEat) {
    PrintSection("Fairness_AllPhilosophersEat",
                 "Every philosopher should eat at least once. A biased scheduler "
                 "could starve some philosophers; the monitor should prevent that.");

    constexpr std::size_t kPhilosophers = 5;
    constexpr int kRounds = 20;

    DiningPhilosophers dp(kPhilosophers);
    std::vector<std::atomic<int>> meals_per_philosopher(kPhilosophers);
    for (auto& m : meals_per_philosopher) { m.store(0); }

    {
        std::vector<std::jthread> philosophers;
        philosophers.reserve(kPhilosophers);
        for (std::size_t id = 0; id < kPhilosophers; ++id) {
            philosophers.emplace_back([&, id] {
                for (int r = 0; r < kRounds; ++r) {
                    dp.PickupForks(id);
                    meals_per_philosopher[id].fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    dp.PutdownForks(id);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
        }
    }

    std::cout << "  Meals per philosopher: ";
    for (std::size_t i = 0; i < kPhilosophers; ++i) {
        int count = meals_per_philosopher[i].load();
        std::cout << "[" << i << "]=" << count << "  ";
        EXPECT_EQ(count, kRounds)
            << "Philosopher " << i << " ate " << count
            << " times, expected " << kRounds << ".";
    }
    std::cout << "\n";
}
