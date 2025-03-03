/**
 * @file StlAlgorithm.hpp
 * @brief Custom implementations of STL algorithm functions
 * @details Hand-rolled versions of any_of, none_of, find_if, find_if_not,
 *          count_if, remove_if, partition, and is_partitioned.
 */

#pragma once

#include <algorithm>
#include <iterator>

namespace StlAlgorithm {

template <typename Iterator, typename UnaryPredicate>
bool MyAnyOf(Iterator first, Iterator last, UnaryPredicate pred) {
    for (; first != last; ++first) {
        if (pred(*first)) return true;
    }
    return false;
}

template <typename Iterator, typename UnaryPredicate>
bool MyNoneOf(Iterator first, Iterator last, UnaryPredicate pred) {
    for (; first != last; ++first) {
        if (pred(*first)) return false;
    }
    return true;
}

template <typename Iterator, typename UnaryPredicate>
Iterator MyFindIf(Iterator first, Iterator last, UnaryPredicate pred) {
    for (; first != last; ++first) {
        if (pred(*first)) return first;
    }
    return last;
}

template <typename Iterator, typename UnaryPredicate>
Iterator MyFindIfNot(Iterator first, Iterator last, UnaryPredicate pred) {
    for (; first != last; ++first) {
        if (!pred(*first)) return first;
    }
    return last;
}

template <typename Iterator, typename UnaryPredicate>
int MyCountIf(Iterator first, Iterator last, UnaryPredicate pred) {
    int count = 0;
    for (; first != last; ++first) {
        if (pred(*first)) ++count;
    }
    return count;
}

template <typename Iterator, typename UnaryPredicate>
Iterator MyRemoveIf(Iterator first, Iterator last, UnaryPredicate pred) {
    Iterator result = first;
    for (; first != last; ++first) {
        if (!pred(*first)) {
            *result = std::move(*first);
            ++result;
        }
    }
    return result;
}

template <typename Iterator, typename UnaryPredicate>
Iterator MyPartition(Iterator first, Iterator last, UnaryPredicate pred) {
    while (first != last) {
        while (first != last && pred(*first)) ++first;
        do { --last; } while (first != last && !pred(*last));
        if (first != last) {
            std::iter_swap(first, last);
            ++first;
        }
    }
    return first;
}

template <typename Iterator, typename UnaryPredicate>
bool MyIsPartitioned(Iterator first, Iterator last, UnaryPredicate pred) {
    for (; first != last; ++first) {
        if (!pred(*first)) break;
    }
    for (; first != last; ++first) {
        if (pred(*first)) return false;
    }
    return true;
}

} // namespace StlAlgorithm
