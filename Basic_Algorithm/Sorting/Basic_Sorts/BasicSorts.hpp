/**
 * @file BasicSorts.hpp
 * @brief O(n^2) comparison sorts: bubble, insertion, selection
 * @details Each sort takes a pair of iterators [first, last) and an optional
 *          comparator. Defaults to ascending order (std::less).
 */

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>

namespace BasicSorts {

template <typename RandomIt, typename Compare>
void BubbleSort(RandomIt first, RandomIt last, Compare comp) {
    if (first == last) return;
    bool swapped = true;
    for (auto end = last; swapped && end != first; --end) {
        swapped = false;
        for (auto it = first; it + 1 != end; ++it) {
            if (comp(*(it + 1), *it)) {
                std::iter_swap(it, it + 1);
                swapped = true;
            }
        }
    }
}

template <typename RandomIt>
void BubbleSort(RandomIt first, RandomIt last) {
    BubbleSort(first, last,
               std::less<typename std::iterator_traits<RandomIt>::value_type>{});
}

template <typename RandomIt, typename Compare>
void InsertionSort(RandomIt first, RandomIt last, Compare comp) {
    if (first == last) return;
    for (auto it = first + 1; it != last; ++it) {
        auto key = std::move(*it);
        auto hole = it;
        while (hole != first && comp(key, *(hole - 1))) {
            *hole = std::move(*(hole - 1));
            --hole;
        }
        *hole = std::move(key);
    }
}

template <typename RandomIt>
void InsertionSort(RandomIt first, RandomIt last) {
    InsertionSort(first, last,
                  std::less<typename std::iterator_traits<RandomIt>::value_type>{});
}

template <typename RandomIt, typename Compare>
void SelectionSort(RandomIt first, RandomIt last, Compare comp) {
    for (auto it = first; it != last; ++it) {
        auto minIt = it;
        for (auto j = it + 1; j != last; ++j) {
            if (comp(*j, *minIt)) minIt = j;
        }
        if (minIt != it) std::iter_swap(it, minIt);
    }
}

template <typename RandomIt>
void SelectionSort(RandomIt first, RandomIt last) {
    SelectionSort(first, last,
                  std::less<typename std::iterator_traits<RandomIt>::value_type>{});
}

} // namespace BasicSorts
