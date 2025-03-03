/**
 * @file DecisionTreeConfig.hpp
 * @brief Configuration structs, SplitCriterion concept, and concrete criterion
 *        implementations for the DecisionTree.
 *
 * @details Defines:
 *   - SplitCriterion concept: any type that provides Impurity(labels) -> double.
 *   - GiniCriterion, EntropyCriterion, MseCriterion: concrete criteria.
 *   - SplitCriterionType enum: used by the legacy Config path.
 *   - Config struct: hyperparameters for tree construction.
 */
#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <concepts>
#include <type_traits>

namespace Ml {

// ─────────────────────────────────────────────────────────────────────────────
// SplitCriterion concept
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Concept that any split-quality metric must satisfy.
 *
 * A SplitCriterion is a callable object whose Impurity() method takes a
 * vector of integer-like labels and returns a double measuring how "mixed"
 * those labels are.  Lower values mean purer nodes.
 *
 * @tparam T The criterion type to check.
 */
template<typename T>
concept SplitCriterion = requires(T criterion, const std::vector<int>& labels) {
    { criterion.Impurity(labels) } -> std::convertible_to<double>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Concrete criterion implementations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Gini impurity: 1 - sum(p_i^2).
 *
 * Measures the probability of misclassifying a randomly chosen element.
 * - Pure node  -> 0.0
 * - Worst case -> 1 - 1/k  (k classes equally represented)
 */
struct GiniCriterion {
    /**
     * @brief Compute Gini impurity for a label vector.
     * @tparam Label Arithmetic type for labels.
     * @param labels The label vector.
     * @return Gini impurity in [0, 1).
     */
    template<typename Label>
    double Impurity(const std::vector<Label>& labels) const {
        if (labels.empty()) return 0.0;
        std::unordered_map<Label, size_t> counts;
        for (const auto& v : labels) counts[v]++;
        double gini = 1.0;
        double n = static_cast<double>(labels.size());
        for (const auto& [key, cnt] : counts) {
            double p = static_cast<double>(cnt) / n;
            gini -= p * p;
        }
        return gini;
    }
};

/**
 * @brief Entropy (information gain): -sum(p_i * log2(p_i)).
 *
 * Measures the average surprise of a random draw from the distribution.
 * - Pure node  -> 0.0
 * - Worst case -> log2(k)  (k classes equally represented)
 */
struct EntropyCriterion {
    /**
     * @brief Compute entropy for a label vector.
     * @tparam Label Arithmetic type for labels.
     * @param labels The label vector.
     * @return Entropy in [0, log2(k)].
     */
    template<typename Label>
    double Impurity(const std::vector<Label>& labels) const {
        if (labels.empty()) return 0.0;
        std::unordered_map<Label, size_t> counts;
        for (const auto& v : labels) counts[v]++;
        double entropy = 0.0;
        double n = static_cast<double>(labels.size());
        for (const auto& [key, cnt] : counts) {
            double p = static_cast<double>(cnt) / n;
            if (p > 0.0) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    }
};

/**
 * @brief Mean Squared Error: (1/n) * sum((y_i - mean)^2).
 *
 * Used for regression trees.  Measures how far each value lies from the
 * node's mean prediction.
 * - Perfect fit -> 0.0
 */
struct MseCriterion {
    /**
     * @brief Compute MSE for a label vector.
     * @tparam Label Arithmetic type for labels.
     * @param labels The label vector.
     * @return MSE >= 0.
     */
    template<typename Label>
    double Impurity(const std::vector<Label>& labels) const {
        if (labels.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& v : labels) sum += static_cast<double>(v);
        double mean = sum / static_cast<double>(labels.size());
        double mse = 0.0;
        for (const auto& v : labels) {
            double diff = static_cast<double>(v) - mean;
            mse += diff * diff;
        }
        return mse / static_cast<double>(labels.size());
    }
};

// Compile-time verification that the concrete types satisfy the concept.
static_assert(SplitCriterion<GiniCriterion>,    "GiniCriterion must satisfy SplitCriterion");
static_assert(SplitCriterion<EntropyCriterion>, "EntropyCriterion must satisfy SplitCriterion");
static_assert(SplitCriterion<MseCriterion>,     "MseCriterion must satisfy SplitCriterion");

// ─────────────────────────────────────────────────────────────────────────────
// Enum and Config (legacy / convenient interface)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Enum selecting a built-in splitting criterion.
 *
 * Used by Config so that callers don't need to spell out concept-satisfying
 * types — just pick one of these names.
 */
enum class SplitCriterionType {
    Gini,       ///< Gini impurity (default for classification)
    Entropy,    ///< Shannon entropy (alternative for classification)
    MSE         ///< Mean squared error (for regression)
};

/**
 * @brief Type trait to check if a type is numeric (arithmetic).
 * @tparam T The type to check.
 */
template<typename T>
struct IsNumeric : std::is_arithmetic<T> {};

/**
 * @brief Helper variable template for IsNumeric.
 * @tparam T The type to check.
 */
template<typename T>
constexpr bool IsNumericV = IsNumeric<T>::value;

/**
 * @brief Configuration struct for tree hyperparameters.
 *
 * All fields have sensible defaults.  Override only what you need.
 *
 * @tparam T Numeric type for features/labels (must be arithmetic).
 */
template<typename T = double>
struct DecisionTreeConfig {
    SplitCriterionType Criterion = SplitCriterionType::Gini;  ///< Splitting criterion
    size_t MaxDepth          = 0;     ///< Max depth (0 = unlimited)
    size_t MinSamplesSplit   = 2;     ///< Min samples to split an internal node
    size_t MinSamplesLeaf    = 1;     ///< Min samples in a leaf node
    size_t MaxFeatures       = 0;     ///< Max features to consider (0 = all)
    T      MinImpurityDecrease = T{0}; ///< Min impurity decrease for splitting
    bool   RandomState       = false;  ///< Whether to use a random seed
    unsigned int Seed        = 42;     ///< Random seed
};

} // namespace Ml
