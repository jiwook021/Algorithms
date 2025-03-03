/**
 * @file QuickDelete.hpp
 * @brief O(1) element removal from unordered vectors
 * @details Swaps the target element with the last element and pops, achieving
 *          constant-time removal at the cost of not preserving element order.
 */

#pragma once

#include <algorithm>
#include <vector>

/**
 * @brief Remove element at index @p idx from an unordered vector in O(1).
 *
 * Replaces the element at @p idx with the last element, then pops the back.
 * Does nothing if @p idx is out of bounds.
 *
 * @tparam T Element type.
 * @param v  The vector to modify.
 * @param idx Index of the element to remove (0-based).
 */
template <typename T>
void QuickRemoveAt(std::vector<T>& v, std::size_t idx) {
    if (idx < v.size()) {
        v.at(idx) = std::move(v.back());
        v.pop_back();
    }
}

/**
 * @brief Remove element at iterator position from an unordered vector in O(1).
 *
 * Replaces the element at @p it with the last element, then pops the back.
 * Does nothing if @p it equals end().
 *
 * @tparam T Element type.
 * @param v  The vector to modify.
 * @param it Iterator to the element to remove.
 */
template <typename T>
void QuickRemoveAt(std::vector<T>& v, typename std::vector<T>::iterator it) {
    if (it != std::end(v)) {
        *it = std::move(v.back());
        v.pop_back();
    }
}
