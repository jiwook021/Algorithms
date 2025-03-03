/**
 * @file DiningPhilosophers.cpp
 * @brief Implementation of the DiningPhilosophers monitor.
 *
 * @details
 * Demonstrates two locking idioms side-by-side:
 *   - std::scoped_lock  -- simpler RAII lock for sections that don't wait.
 *   - std::unique_lock   -- required by condition_variable::wait().
 */

#include "DiningPhilosophers.hpp"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DiningPhilosophers::DiningPhilosophers(std::size_t num_philosophers)
    : num_philosophers_(num_philosophers),
      states_(num_philosophers, State::Thinking),
      philosopher_cv_(num_philosophers) {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

DiningPhilosophers::State
DiningPhilosophers::GetState(std::size_t philosopher_id) const {
    // scoped_lock: simple critical section, no CV waiting needed.
    std::scoped_lock lock(monitor_mutex_);
    return states_[philosopher_id];
}

void DiningPhilosophers::PickupForks(std::size_t philosopher_id) {
    // unique_lock: condition_variable::wait() requires it.
    std::unique_lock lock(monitor_mutex_);
    states_[philosopher_id] = State::Hungry;
    Test(philosopher_id);
    philosopher_cv_[philosopher_id].wait(lock, [&] {
        return states_[philosopher_id] == State::Eating;
    });
}

void DiningPhilosophers::PutdownForks(std::size_t philosopher_id) {
    // scoped_lock: straightforward critical section -- release forks, wake neighbors.
    std::scoped_lock lock(monitor_mutex_);
    states_[philosopher_id] = State::Thinking;

    std::size_t left  = (philosopher_id + num_philosophers_ - 1) % num_philosophers_;
    std::size_t right = (philosopher_id + 1) % num_philosophers_;
    Test(left);
    Test(right);
}

bool DiningPhilosophers::CheckNoAdjacentEating() const {
    // scoped_lock: snapshot all states atomically, then check the invariant.
    std::scoped_lock lock(monitor_mutex_);
    for (std::size_t i = 0; i < num_philosophers_; ++i) {
        std::size_t next = (i + 1) % num_philosophers_;
        if (states_[i] == State::Eating && states_[next] == State::Eating) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Private helpers  (caller must hold monitor_mutex_)
// ---------------------------------------------------------------------------

void DiningPhilosophers::Test(std::size_t philosopher_id) {
    std::size_t left  = (philosopher_id + num_philosophers_ - 1) % num_philosophers_;
    std::size_t right = (philosopher_id + 1) % num_philosophers_;

    if (states_[philosopher_id] == State::Hungry &&
        states_[left]  != State::Eating &&
        states_[right] != State::Eating) {
        states_[philosopher_id] = State::Eating;
        philosopher_cv_[philosopher_id].notify_one();
    }
}
