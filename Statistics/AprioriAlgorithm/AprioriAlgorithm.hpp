/**
 * @file AprioriAlgorithm.hpp
 * @brief Apriori Algorithm for association rule mining.
 *
 * @details
 * The Apriori algorithm discovers frequent itemsets in transactional databases
 * and generates association rules that satisfy minimum support and confidence
 * thresholds.
 *
 * Key concepts:
 *   - Support(X) = count(X) / |T|, where |T| is total transactions
 *   - Confidence(X -> Y) = Support(X U Y) / Support(X)
 *   - Apriori property: all subsets of a frequent itemset must be frequent
 */

#ifndef APRIORI_ALGORITHM_HPP
#define APRIORI_ALGORITHM_HPP

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Represents a single association rule with confidence and support.
 */
struct AssociationRule {
    std::string Antecedent;   ///< Left-hand side item
    std::string Consequent;   ///< Right-hand side item
    double Confidence;        ///< Confidence of the rule
    int SupportCount;         ///< Number of transactions containing both items
};

/**
 * @brief Holds the complete result of an Apriori algorithm run.
 */
struct AprioriResult {
    std::unordered_map<std::string, int> ItemCounts;   ///< Support count for each item
    std::set<std::string> FrequentItems;                ///< Items meeting min support
    std::unordered_map<std::string, int> PairCounts;    ///< Support count for item pairs
    std::vector<AssociationRule> Rules;                  ///< Generated association rules
};

/**
 * @class AprioriAlgorithm
 * @brief Implements the Apriori algorithm for mining association rules.
 *
 * The algorithm proceeds in two phases:
 *   1. Frequent itemset generation: find all itemsets with support >= minSupport
 *   2. Rule generation: from frequent itemsets, generate rules with
 *      confidence >= minConfidence
 */
class AprioriAlgorithm {
public:
    using Transaction = std::vector<std::string>;
    using Transactions = std::vector<Transaction>;

    /**
     * @brief Generate all candidate pairs from a set of frequent items.
     * @param items Set of frequent items.
     * @return Vector of candidate pairs.
     */
    std::vector<std::pair<std::string, std::string>>
    GenerateCandidates(const std::set<std::string>& items) const;

    /**
     * @brief Count how many transactions contain all items in the given itemset.
     * @param transactionsData The full set of transactions.
     * @param itemset The itemset to search for.
     * @return Number of transactions containing the itemset.
     */
    int CountOccurrences(const Transactions& transactionsData,
                         const std::set<std::string>& itemset) const;

    /**
     * @brief Run the Apriori algorithm on the given transactions.
     * @param transactionsData The transactional dataset.
     * @param minSupport Minimum support count threshold.
     * @param minConfidence Minimum confidence threshold (0.0 to 1.0).
     * @return AprioriResult containing frequent items, pairs, and rules.
     */
    AprioriResult Run(const Transactions& transactionsData,
                      int minSupport, double minConfidence) const;
};

#endif // APRIORI_ALGORITHM_HPP
