/**
 * @file PriorityQueue.cpp
 * @brief PriorityQueue binary-heap implementation.
 */

#include "PriorityQueue.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
constexpr const char* kEmptyQueueMessage = "PriorityQueue is empty";
}

template <typename T, typename Container, typename Compare>
PriorityQueue<T, Container, Compare>::PriorityQueue(
    const Compare& cmp, const Container& values)
    : heap_(values), comp_(cmp) {
    BuildHeap();
}

template <typename T, typename Container, typename Compare>
PriorityQueue<T, Container, Compare>::PriorityQueue(
    typename Container::const_iterator first,
    typename Container::const_iterator last,
    const Compare& cmp)
    : heap_(first, last), comp_(cmp) {
    BuildHeap();
}

template <typename T, typename Container, typename Compare>
bool PriorityQueue<T, Container, Compare>::Empty() const noexcept {
    return heap_.empty();
}

template <typename T, typename Container, typename Compare>
std::size_t PriorityQueue<T, Container, Compare>::Size() const noexcept {
    return heap_.size();
}

template <typename T, typename Container, typename Compare>
const typename PriorityQueue<T, Container, Compare>::ValueType&
PriorityQueue<T, Container, Compare>::Top() const {
    if (Empty()) {
        throw std::out_of_range(kEmptyQueueMessage);
    }

    return heap_.front();
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::Push(const ValueType& value) {
    heap_.push_back(value);
    HeapifyUp(heap_.size() - 1);
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::Push(ValueType&& value) {
    heap_.push_back(std::move(value));
    HeapifyUp(heap_.size() - 1);
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::Pop() {
    if (Empty()) {
        throw std::out_of_range(kEmptyQueueMessage);
    }

    if (heap_.size() == 1) {
        heap_.pop_back();
        return;
    }

    heap_.front() = std::move(heap_.back());
    heap_.pop_back();
    HeapifyDown(0);
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::Clear() noexcept {
    heap_.clear();
}

template <typename T, typename Container, typename Compare>
std::size_t PriorityQueue<T, Container, Compare>::Parent(std::size_t index) {
    return (index - 1) / 2;
}

template <typename T, typename Container, typename Compare>
std::size_t PriorityQueue<T, Container, Compare>::Left(std::size_t index) {
    return (2 * index) + 1;
}

template <typename T, typename Container, typename Compare>
std::size_t PriorityQueue<T, Container, Compare>::Right(std::size_t index) {
    return (2 * index) + 2;
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::HeapifyDown(std::size_t index) {
    while (true) {
        std::size_t best = index;
        const std::size_t left = Left(index);
        const std::size_t right = Right(index);

        if (left < heap_.size() && HasHigherPriority(left, best)) {
            best = left;
        }

        if (right < heap_.size() && HasHigherPriority(right, best)) {
            best = right;
        }

        if (best == index) {
            return;
        }

        std::swap(heap_[index], heap_[best]);
        index = best;
    }
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::HeapifyUp(std::size_t index) {
    while (index > 0) {
        const std::size_t parent = Parent(index);

        if (!HasHigherPriority(index, parent)) {
            return;
        }

        std::swap(heap_[index], heap_[parent]);
        index = parent;
    }
}

template <typename T, typename Container, typename Compare>
void PriorityQueue<T, Container, Compare>::BuildHeap() {
    for (std::size_t index = heap_.size() / 2; index > 0; --index) {
        HeapifyDown(index - 1);
    }
}

template <typename T, typename Container, typename Compare>
bool PriorityQueue<T, Container, Compare>::HasHigherPriority(
    std::size_t candidate,
    std::size_t current) const {
    return comp_(heap_[current], heap_[candidate]);
}

template <typename T, typename Container>
MinPriorityQueue<T, Container>::MinPriorityQueue(
    const std::greater<T>& cmp,
    const Container& values)
    : PriorityQueue<T, Container, std::greater<T>>(cmp, values) {}

template <typename T, typename Container>
MinPriorityQueue<T, Container>::MinPriorityQueue(
    typename Container::const_iterator first,
    typename Container::const_iterator last,
    const std::greater<T>& cmp)
    : PriorityQueue<T, Container, std::greater<T>>(first, last, cmp) {}

template <typename T, typename Container>
MaxPriorityQueue<T, Container>::MaxPriorityQueue(
    const std::less<T>& cmp,
    const Container& values)
    : PriorityQueue<T, Container, std::less<T>>(cmp, values) {}

template <typename T, typename Container>
MaxPriorityQueue<T, Container>::MaxPriorityQueue(
    typename Container::const_iterator first,
    typename Container::const_iterator last,
    const std::less<T>& cmp)
    : PriorityQueue<T, Container, std::less<T>>(first, last, cmp) {}

template class PriorityQueue<int, std::vector<int>, std::less<int>>;
template class PriorityQueue<int, std::vector<int>, std::greater<int>>;
template class PriorityQueue<double, std::vector<double>, std::less<double>>;
template class PriorityQueue<double, std::vector<double>, std::greater<double>>;

template class MinPriorityQueue<int>;
template class MaxPriorityQueue<int>;
template class MinPriorityQueue<double>;
template class MaxPriorityQueue<double>;
