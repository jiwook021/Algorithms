/**
 * @file Tuples.hpp
 * @brief Custom variadic tuple utilities
 * @details Tuple printing, zip, and summary statistics using variadic templates,
 *          fold expressions, and std::apply. Demonstrates C++17 parameter pack techniques.
 */

#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

/**
 * @brief Print a comma-separated list of arguments to the given stream.
 * @tparam T First argument type.
 * @tparam Ts Remaining argument types.
 */
template <typename T, typename... Ts>
void PrintArgs(std::ostream& os, const T& v, const Ts&... vs) {
    os << v;
    (void)std::initializer_list<int>{((os << ", " << vs), 0)...};
}

/**
 * @brief Stream insertion operator for std::tuple.
 *
 * Prints the tuple as (elem1, elem2, ...).
 */
template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    auto printToOs = [&os](const auto&... xs) {
        PrintArgs(os, xs...);
    };
    os << "(";
    std::apply(printToOs, t);
    return os << ")";
}

/**
 * @brief Compute sum, min, max, and average of a range.
 * @tparam T Range type (must support std::begin/end and size()).
 * @return Tuple of (sum, min, max, average) as doubles.
 */
template <typename T>
std::tuple<double, double, double, double> SumMinMaxAvg(const T& range) {
    auto minMax = std::minmax_element(std::begin(range), std::end(range));
    auto sum = std::accumulate(std::begin(range), std::end(range), 0.0);
    return {sum, *minMax.first, *minMax.second, sum / range.size()};
}

/**
 * @brief Interleave two tuples element-wise: zip((a,b), (x,y)) = (a,x,b,y).
 * @tparam T1 First tuple type.
 * @tparam T2 Second tuple type.
 */
template <typename T1, typename T2>
auto Zip(const T1& a, const T2& b) {
    auto z = [](auto... xs) {
        return [xs...](auto... ys) {
            return std::tuple_cat(std::make_tuple(xs, ys)...);
        };
    };
    return std::apply(std::apply(z, a), b);
}
