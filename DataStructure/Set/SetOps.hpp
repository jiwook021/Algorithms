/**
 * @file SetOps.hpp
 * @brief Set operations utility functions using std::set and std::multiset
 * @details Demonstrates union, intersection, and difference operations on
 *          STL ordered sets. Showcases comparator usage and iterator patterns.
 */

#pragma once

#include <algorithm>
#include <set>

/**
 * @brief Compute the union of two sets (St1 U St2) and store in result.
 * @tparam T Element type.
 * @param st1 First set.
 * @param st2 Second set.
 * @param result Output set containing all elements from both inputs.
 *
 * Time: O(n + m) where n = |st1|, m = |st2|.
 */
template <typename T>
void SetUnion(const std::set<T>& st1, const std::set<T>& st2, std::set<T>& result) {
    result = st1;
    for (const auto& elem : st2) {
        result.insert(elem);
    }
}

/**
 * @brief Compute the intersection of two sets and store in result.
 * @tparam T Element type.
 * @param st1 First set.
 * @param st2 Second set.
 * @param result Output set containing only elements present in both inputs.
 *
 * Time: O(n + m).
 */
template <typename T>
void SetIntersection(const std::set<T>& st1, const std::set<T>& st2, std::set<T>& result) {
    result.clear();
    for (const auto& elem : st1) {
        if (st2.count(elem) > 0) {
            result.insert(elem);
        }
    }
}

/**
 * @brief Compute the difference (st1 - st2) and store in result.
 * @tparam T Element type.
 * @param st1 First set.
 * @param st2 Second set.
 * @param result Output set containing elements in st1 but not in st2.
 *
 * Time: O(n log m).
 */
template <typename T>
void SetDifference(const std::set<T>& st1, const std::set<T>& st2, std::set<T>& result) {
    result.clear();
    for (const auto& elem : st1) {
        if (st2.count(elem) == 0) {
            result.insert(elem);
        }
    }
}
