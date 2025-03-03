/**
 * @file quick_sort.hpp
 * @brief Quick sort — the most widely-used general-purpose sorting algorithm.
 *
 * How it works (high level):
 *   1. Pick a "pivot" element from the array.
 *   2. Partition: move everything smaller than pivot to the left,
 *      everything larger to the right.  The pivot is now in its final spot.
 *   3. Recursively sort the left half and right half.
 *
 * This implementation adds two optimisations used in real-world libraries:
 *   - Median-of-three pivot: picks the median of the first, middle, and last
 *     elements to avoid worst-case O(n²) on already-sorted input.
 *   - Insertion sort fallback: for small partitions (≤ 16 elements),
 *     insertion sort is faster due to lower overhead.
 *
 * Complexity:
 *   Average / best: O(n log n)
 *   Worst (rare with median-of-three): O(n²)
 *   Space: O(log n) stack frames.
 */

#pragma once

#include <algorithm>    // std::iter_swap
#include <cstddef>
#include <functional>   // std::less
#include <iterator>     // std::iterator_traits — extracts element type from an iterator
#include <type_traits>  // std::is_same_v

// Internal helpers — not meant to be called directly.
namespace SortImpl {

/**
 * Insertion sort for small sub-arrays.
 * When the array is tiny (≤ 16), the overhead of recursion + partitioning
 * in quick sort is worse than just doing simple insertion sort.
 */
template <typename RandomIt, typename Compare>
void InsertionSort(RandomIt first, RandomIt Last, Compare Comp) {
    for (auto It = first + 1; It < Last; ++It) {
        auto key = std::move(*It);
        auto Hole = It;
        while (Hole > first && Comp(key, *(Hole - 1))) {
            *Hole = std::move(*(Hole - 1));
            --Hole;
        }
        *Hole = std::move(key);
    }
}

/**
 * Pick a good pivot using median-of-three.
 *
 * Compares the first, middle, and last elements, sorts those three,
 * and returns an iterator to the median (placed at the last position).
 *
 * Why? If the array is already sorted, picking the first element as pivot
 * gives O(n²).  Picking the median of three avoids that in most cases.
 */
template <typename RandomIt, typename Compare>
RandomIt MedianOfThree(RandomIt first, RandomIt Last, Compare Comp) {
    auto Mid = first + (Last - first) / 2;  // middle element
    auto back = Last - 1;                    // last element
    // Sort the three elements: first, mid, back
    if (Comp(*Mid, *first)) std::iter_swap(first, Mid);
    if (Comp(*back, *first)) std::iter_swap(first, back);
    if (Comp(*Mid, *back)) std::iter_swap(Mid, back);
    // Now *back is the median of the three → use it as pivot
    return back;
}

/**
 * Partition the range [first, last) around a pivot.
 *
 * After partitioning:
 *   - Elements in [first, pivot_pos) are all < pivot
 *   - *pivot_pos == pivot
 *   - Elements in (pivot_pos, last) are all >= pivot
 *
 * Returns: iterator pointing to the pivot's final position.
 */
template <typename RandomIt, typename Compare>
RandomIt Partition(RandomIt first, RandomIt Last, Compare Comp) {
    // Step 1: Choose pivot via median-of-three, now sitting at *(last-1)
    auto PivotIt = MedianOfThree(first, Last, Comp);
    auto Pivot = std::move(*PivotIt);
    std::iter_swap(PivotIt, Last - 1);   // stash pivot at the end

    // Step 2: Walk through [first, last-1), moving small elements to the left
    auto Store = first;  // boundary: everything left of 'store' is < pivot
    for (auto It = first; It < Last - 1; ++It) {
        if (Comp(*It, Pivot)) {
            std::iter_swap(It, Store);
            ++Store;
        }
    }

    // Step 3: Place pivot in its final sorted position
    std::iter_swap(Store, Last - 1);
    return Store;
}

} // namespace sort_impl

/**
 * @brief Sort range [first, last) using quick sort with comparator @p comp.
 *
 * Usage:
 *   std::vector<int> v{5, 3, 1, 4, 2};
 *   quick_sort(v.begin(), v.end());                        // ascending
 *   quick_sort(v.begin(), v.end(), std::greater<int>{});   // descending
 */
template <typename RandomIt, typename Compare>
void QuickSort(RandomIt first, RandomIt Last, Compare Comp) {
    // Compile-time check: quick sort needs random-access iterators (like vector),
    // not forward-only iterators (like linked list).
    static_assert(
        std::is_same_v<typename std::iterator_traits<RandomIt>::iterator_category,
                       std::random_access_iterator_tag>,
        "quick_sort requires random access iterators");

    constexpr std::ptrdiff_t KThreshold = 16;  // switch to insertion sort below this size

    // Main loop: keep partitioning until sub-array is small enough
    while (Last - first > KThreshold) {
        auto Pivot = SortImpl::Partition(first, Last, Comp);

        // Optimisation: recurse on the SMALLER half, loop on the LARGER half.
        // This limits the call stack depth to O(log n) instead of O(n).
        if (Pivot - first < Last - (Pivot + 1)) {
            QuickSort(first, Pivot, Comp);   // sort left (smaller)
            first = Pivot + 1;                // continue with right (larger) in this loop
        } else {
            QuickSort(Pivot + 1, Last, Comp); // sort right (smaller)
            Last = Pivot;                      // continue with left (larger) in this loop
        }
    }

    // Small sub-array: finish with insertion sort (faster for small n)
    SortImpl::InsertionSort(first, Last, Comp);
}

/**
 * Convenience overload: sorts ascending by default.
 *
 * std::iterator_traits<RandomIt>::value_type extracts the element type from
 * the iterator.  E.g. for vector<int>::iterator → int, so std::less<int>{}.
 */
template <typename RandomIt>
void QuickSort(RandomIt first, RandomIt Last) {
    QuickSort(first, Last,
               std::less<typename std::iterator_traits<RandomIt>::value_type>{});
}
