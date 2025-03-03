/**
 * @file Norm.hpp
 * @brief Computation of p-norms for vectors.
 * @details Supports L1, L2, general Lp, and L-infinity norms on
 *          any range of floating-point values, constrained by the
 *          VectorLike concept.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numeric>
#include <ranges>
#include <stdexcept>

/**
 * @brief Concept requiring a range whose elements are floating-point.
 */
template <typename C>
concept VectorLike = std::ranges::range<C> &&
    std::floating_point<std::ranges::range_value_t<C>>;

/**
 * @brief Vector p-norm calculator.
 *
 * @details Encapsulates the computation of the p-norm:
 *          ||x||_p = (sum |x_i|^p)^{1/p}  for finite p >= 1
 *          ||x||_inf = max |x_i|            for p = infinity
 *
 *          All public methods are constrained by the VectorLike concept
 *          and implemented with std::ranges algorithms where appropriate.
 */
class Norm {
public:
    /**
     * @brief Compute the L1 (Manhattan) norm.
     * @tparam Container  Any VectorLike range.
     * @param vec  The container of floating-point values.
     * @return The L1 norm.
     *
     * @details Time O(n), Space O(1).
     */
    template <VectorLike Container>
    static auto L1(const Container& vec) {
        using T = std::ranges::range_value_t<Container>;
        if (std::ranges::empty(vec)) return T{0};

        auto abs_view = vec | std::views::transform(
            [](T x) { return std::abs(x); });
        return std::accumulate(std::ranges::begin(abs_view),
                               std::ranges::end(abs_view), T{0});
    }

    /**
     * @brief Compute the L2 (Euclidean) norm.
     * @tparam Container  Any VectorLike range.
     * @param vec  The container of floating-point values.
     * @return The L2 norm.
     *
     * @details Time O(n), Space O(1).
     */
    template <VectorLike Container>
    static auto L2(const Container& vec) {
        using T = std::ranges::range_value_t<Container>;
        if (std::ranges::empty(vec)) return T{0};

        auto sq_view = vec | std::views::transform(
            [](T x) { return x * x; });
        T sum_sq = std::accumulate(std::ranges::begin(sq_view),
                                   std::ranges::end(sq_view), T{0});
        return static_cast<T>(std::sqrt(sum_sq));
    }

    /**
     * @brief Compute the L-infinity (max absolute value) norm.
     * @tparam Container  Any VectorLike range.
     * @param vec  The container of floating-point values.
     * @return The L-infinity norm.
     *
     * @details Time O(n), Space O(1).
     */
    template <VectorLike Container>
    static auto LInf(const Container& vec) {
        using T = std::ranges::range_value_t<Container>;
        if (std::ranges::empty(vec)) return T{0};

        auto abs_view = vec | std::views::transform(
            [](T x) { return std::abs(x); });
        return *std::ranges::max_element(abs_view);
    }

    /**
     * @brief Compute the general p-norm.
     * @tparam Container  Any VectorLike range.
     * @param vec  The container of floating-point values.
     * @param p    The norm order (>= 1, or infinity).
     * @return The p-norm as the container's value type.
     * @throws std::invalid_argument if p < 1 and is not infinity.
     *
     * @details Time O(n), Space O(1).
     *          Delegates to L1, L2, or LInf when p matches exactly.
     */
    template <VectorLike Container>
    static auto PNorm(const Container& vec, double p) {
        using T = std::ranges::range_value_t<Container>;
        if (std::ranges::empty(vec)) return T{0};

        if (!std::isinf(p) && p < 1.0) {
            throw std::invalid_argument(
                "The norm order p must be >= 1 or infinity.");
        }

        if (std::isinf(p)) return LInf(vec);
        if (p == 1.0)      return L1(vec);
        if (p == 2.0)      return L2(vec);

        auto pow_view = vec | std::views::transform(
            [p](T x) -> T { return static_cast<T>(std::pow(std::abs(x), p)); });
        T sum_pow = std::accumulate(std::ranges::begin(pow_view),
                                    std::ranges::end(pow_view), T{0});
        return static_cast<T>(std::pow(sum_pow, 1.0 / p));
    }
};
