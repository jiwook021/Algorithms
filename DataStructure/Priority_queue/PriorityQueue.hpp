/**
 * @file PriorityQueue.hpp
 * @brief Custom priority queue (binary heap) implementation
 * @details Declares a binary-heap priority queue backed by a contiguous
 *          container. The default comparator builds a max-heap; std::greater
 *          builds a min-heap. Template definitions are compiled from
 *          PriorityQueue.cpp through explicit instantiations.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

/**
 * @class PriorityQueue
 * @tparam T Element type.
 * @tparam Container Underlying container (default: std::vector<T>).
 * @tparam Compare Comparator (default: std::less -- max-heap).
 *
 * Uses std::less for max-heap: comp(parent, child) == true means
 * parent has LOWER priority than child, triggering a swap.
 *
 * Complexity:
 *   Push : O(log n)
 *   Pop  : O(log n)
 *   Top  : O(1)
 *   Space: O(n)
 */
template <typename T,
          typename Container = std::vector<T>,
          typename Compare = std::less<typename Container::value_type>>
class PriorityQueue {
public:
    typedef typename Container::value_type ValueType;

    explicit PriorityQueue(const Compare& cmp = Compare(),
                           const Container& values = Container());

    PriorityQueue(typename Container::const_iterator first,
                  typename Container::const_iterator last,
                  const Compare& cmp = Compare());

    [[nodiscard]] bool Empty() const noexcept;
    [[nodiscard]] std::size_t Size() const noexcept;
    [[nodiscard]] const ValueType& Top() const;
    void Push(const ValueType& value);
    void Push(ValueType&& value);
    void Pop();
    void Clear() noexcept;

private:
    Container heap_;
    Compare comp_;

    static std::size_t Parent(std::size_t i);
    static std::size_t Left(std::size_t i);
    static std::size_t Right(std::size_t i);

    void HeapifyDown(std::size_t i);
    void HeapifyUp(std::size_t i);
    void BuildHeap();
    [[nodiscard]] bool HasHigherPriority(std::size_t candidate,
                                         std::size_t current) const;
};

/// Min-heap priority queue wrapper.
template <typename T, typename Container = std::vector<T>>
class MinPriorityQueue : public PriorityQueue<T, Container, std::greater<T>> {
public:
    explicit MinPriorityQueue(const std::greater<T>& cmp = std::greater<T>(),
                              const Container& values = Container());

    MinPriorityQueue(typename Container::const_iterator first,
                     typename Container::const_iterator last,
                     const std::greater<T>& cmp = std::greater<T>());
};

/// Max-heap priority queue wrapper.
template <typename T, typename Container = std::vector<T>>
class MaxPriorityQueue : public PriorityQueue<T, Container, std::less<T>> {
public:
    explicit MaxPriorityQueue(const std::less<T>& cmp = std::less<T>(),
                              const Container& values = Container());

    MaxPriorityQueue(typename Container::const_iterator first,
                     typename Container::const_iterator last,
                     const std::less<T>& cmp = std::less<T>());
};
