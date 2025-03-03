/**
 * @file Iterators.hpp
 * @brief Custom iterator and range classes demonstrating iterator protocol
 * @details Implements NumIterator (forward iterator) and NumRange (iterable range)
 *          to demonstrate how C++ range-based for loops work under the hood.
 */

#pragma once

/**
 * @class NumIterator
 * @brief A forward iterator that yields consecutive integers.
 *
 * Satisfies the requirements for a forward iterator:
 *   - Dereferenceable (operator*)
 *   - Incrementable (operator++)
 *   - Comparable (operator!=)
 */
class NumIterator {
public:
    /**
     * @brief Construct an iterator at position @p position.
     * @param position The integer value this iterator points to.
     */
    explicit NumIterator(int position = 0) : position_(position) {}

    /** @brief Dereference: return the current integer value. */
    int operator*() const { return position_; }

    /** @brief Pre-increment: advance to the next integer. */
    NumIterator& operator++() {
        ++position_;
        return *this;
    }

    /** @brief Inequality comparison for range termination. */
    bool operator!=(const NumIterator& other) const {
        return position_ != other.position_;
    }

private:
    int position_;
};

/**
 * @class NumRange
 * @brief An iterable range [from, to) that produces consecutive integers.
 *
 * Provides begin()/end() so it works with range-based for loops:
 *   for (int i : NumRange(1, 10)) { ... }
 */
class NumRange {
public:
    /**
     * @brief Construct a range [from, to).
     * @param from Starting value (inclusive).
     * @param to Ending value (exclusive).
     */
    NumRange(int from, int to) : from_(from), to_(to) {}

    /** @brief Iterator to the first element. */
    NumIterator begin() const { return NumIterator(from_); }

    /** @brief Past-the-end iterator. */
    NumIterator end() const { return NumIterator(to_); }

private:
    int from_;
    int to_;
};
