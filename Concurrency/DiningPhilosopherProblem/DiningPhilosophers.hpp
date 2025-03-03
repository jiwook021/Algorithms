/**
 * @file DiningPhilosophers.hpp
 * @brief Dining Philosophers monitor -- prevents deadlock with centralized arbitration.
 *
 * @details
 * Encapsulates the philosopher-state array, the fork-checking logic, and the
 * condition variables used to wake hungry philosophers. The Test() method
 * acts as a centralized arbiter: a philosopher transitions to Eating only
 * when neither neighbor is eating.
 *
 * Uses std::scoped_lock for RAII-based locking where a simple critical section
 * suffices (GetState, PutdownForks), and std::unique_lock only where
 * condition-variable waiting is required (PickupForks).
 *
 * Complexity:
 *   - PickupForks():  O(1) (may block)
 *   - PutdownForks(): O(1)
 */

#ifndef DINING_PHILOSOPHERS_HPP
#define DINING_PHILOSOPHERS_HPP

#include <cstddef>
#include <mutex>
#include <vector>
#include <condition_variable>

/**
 * @class DiningPhilosophers
 * @brief Monitor-based solution to the Dining Philosophers problem.
 *
 * Uses a centralized arbiter pattern: each philosopher must be Hungry and
 * have no Eating neighbor before transitioning to Eating. Condition variables
 * provide per-philosopher blocking/waking.
 */
class DiningPhilosophers {
public:
    /** @brief Possible states for each philosopher. */
    enum class State { Thinking, Hungry, Eating };

    /**
     * @brief Construct a table with the given number of philosophers.
     * @param num_philosophers Number of philosophers.
     */
    explicit DiningPhilosophers(std::size_t num_philosophers);

    /** @brief Non-copyable. */
    DiningPhilosophers(const DiningPhilosophers&) = delete;
    /** @brief Non-copyable. */
    DiningPhilosophers& operator=(const DiningPhilosophers&) = delete;

    /**
     * @brief Return the number of philosophers at the table.
     * @return Philosopher count.
     */
    std::size_t Count() const { return num_philosophers_; }

    /**
     * @brief Thread-safe state query.  Uses std::scoped_lock (no CV needed).
     * @param philosopher_id Philosopher index.
     * @return Current state of the philosopher.
     */
    State GetState(std::size_t philosopher_id) const;

    /**
     * @brief Blocking call -- returns only when the philosopher is Eating.
     *
     * Uses std::unique_lock because condition_variable::wait requires it.
     * @param philosopher_id Philosopher index.
     */
    void PickupForks(std::size_t philosopher_id);

    /**
     * @brief Release forks and signal neighbors.
     *
     * Demonstrates std::scoped_lock for a straightforward critical section.
     * @param philosopher_id Philosopher index.
     */
    void PutdownForks(std::size_t philosopher_id);

    /**
     * @brief Atomically verify no two adjacent philosophers are Eating.
     *
     * Acquires the monitor lock once, then scans all adjacent pairs.
     * Useful for testing the mutual-exclusion invariant without TOCTOU races.
     * @return True if the invariant holds (no adjacent pair is Eating).
     */
    bool CheckNoAdjacentEating() const;

private:
    /**
     * @brief Attempt to transition philosopher to Eating.
     * @note Caller must hold monitor_mutex_.
     * @param philosopher_id Philosopher index.
     */
    void Test(std::size_t philosopher_id);

    std::size_t num_philosophers_;                      ///< Number of philosophers.
    std::vector<State> states_;                         ///< Per-philosopher state.
    mutable std::mutex monitor_mutex_;                  ///< Global monitor lock.
    std::vector<std::condition_variable> philosopher_cv_; ///< Per-philosopher CV.
};

#endif // DINING_PHILOSOPHERS_HPP
