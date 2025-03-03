/**
 * @file KeepVectorSorted.hpp
 * @brief Sorted insertion into a container
 * @details Template function that inserts an element into a sorted container
 *          while maintaining sort order using std::lower_bound.
 */

#pragma once

#include <algorithm>

namespace KeepVectorSorted {

/**
 * @brief Insert value into sorted container maintaining order.
 * @tparam C Container type (must support begin/end/insert)
 * @tparam T Value type
 */
template <typename C, typename T>
void InsertSorted(C& container, const T& value) {
    auto it = std::lower_bound(container.begin(), container.end(), value);
    container.insert(it, value);
}

} // namespace KeepVectorSorted
