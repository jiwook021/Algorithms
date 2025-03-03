/**
 * @file test_AprioriAlgorithm.cpp
 * @brief Unit tests for the AprioriAlgorithm class.
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <string>

#include "AprioriAlgorithm.hpp"

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_TRUE(cond, msg)                                       \
    do {                                                             \
        if (!(cond)) {                                               \
            std::cerr << "FAIL: " << msg << " (" << __FILE__        \
                      << ":" << __LINE__ << ")\n";                   \
            testsFailed++;                                           \
        } else {                                                     \
            std::cout << "PASS: " << msg << "\n";                    \
            testsPassed++;                                           \
        }                                                            \
    } while (0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)

/**
 * @brief Test candidate generation from a set of 3 items produces 3 pairs.
 */
void TestGenerateCandidatesThreeItems() {
    AprioriAlgorithm algorithm;
    std::set<std::string> items = {"A", "B", "C"};
    auto candidates = algorithm.GenerateCandidates(items);
    ASSERT_EQ(candidates.size(), 3u,
              "GenerateCandidates({A,B,C}) should produce 3 pairs");
}

/**
 * @brief Test candidate generation from a single item produces 0 pairs.
 */
void TestGenerateCandidatesSingleItem() {
    AprioriAlgorithm algorithm;
    std::set<std::string> items = {"A"};
    auto candidates = algorithm.GenerateCandidates(items);
    ASSERT_TRUE(candidates.empty(),
                "GenerateCandidates with 1 item should produce 0 pairs");
}

/**
 * @brief Test candidate generation from empty set.
 */
void TestGenerateCandidatesEmpty() {
    AprioriAlgorithm algorithm;
    std::set<std::string> items;
    auto candidates = algorithm.GenerateCandidates(items);
    ASSERT_TRUE(candidates.empty(),
                "GenerateCandidates with empty set should produce 0 pairs");
}

/**
 * @brief Test CountOccurrences correctly counts itemset appearances.
 */
void TestCountOccurrences() {
    AprioriAlgorithm algorithm;
    AprioriAlgorithm::Transactions trans = {
        {"A", "B", "C"}, {"A", "B"}, {"B", "C"}};
    std::set<std::string> itemset = {"A", "B"};
    int count = algorithm.CountOccurrences(trans, itemset);
    ASSERT_EQ(count, 2, "CountOccurrences({A,B}) in 3 transactions should be 2");
}

/**
 * @brief Test CountOccurrences returns 0 when itemset is not fully present.
 */
void TestCountOccurrencesNone() {
    AprioriAlgorithm algorithm;
    AprioriAlgorithm::Transactions trans = {{"A"}, {"B"}};
    std::set<std::string> itemset = {"A", "B"};
    int count = algorithm.CountOccurrences(trans, itemset);
    ASSERT_EQ(count, 0,
              "CountOccurrences should be 0 when no transaction has both items");
}

/**
 * @brief Test full Run produces correct frequent items, pairs, and rules.
 */
void TestRunProducesRules() {
    AprioriAlgorithm algorithm;
    AprioriAlgorithm::Transactions trans = {
        {"bread", "milk"},
        {"bread", "diapers", "beer"},
        {"milk", "diapers", "beer"},
        {"bread", "milk", "diapers", "beer"}};

    const AprioriResult result = algorithm.Run(trans, 2, 0.5);

    ASSERT_TRUE(!result.FrequentItems.empty(),
                "Run should produce frequent items with minSupport=2");
    ASSERT_TRUE(!result.PairCounts.empty(),
                "Run should produce frequent pairs");
    ASSERT_TRUE(!result.Rules.empty(),
                "Run should produce association rules with minConfidence=0.5");
}

/**
 * @brief Test that higher confidence threshold yields fewer or equal rules.
 */
void TestHighConfidenceReducesRules() {
    AprioriAlgorithm algorithm;
    AprioriAlgorithm::Transactions trans = {
        {"A", "B"}, {"A", "B"}, {"A", "C"}, {"B", "C"}, {"A", "B", "C"}};

    const AprioriResult resultLow = algorithm.Run(trans, 2, 0.3);
    const AprioriResult resultHigh = algorithm.Run(trans, 2, 0.9);

    ASSERT_TRUE(resultHigh.Rules.size() <= resultLow.Rules.size(),
                "Higher confidence threshold should produce fewer or equal rules");
}

/**
 * @brief Test that all generated rules have confidence >= minConfidence.
 */
void TestRuleConfidenceAboveThreshold() {
    AprioriAlgorithm algorithm;
    AprioriAlgorithm::Transactions trans = {
        {"A", "B"}, {"A", "B", "C"}, {"B", "C"}, {"A", "C"}};

    const double MIN_CONF = 0.6;
    const AprioriResult result = algorithm.Run(trans, 2, MIN_CONF);

    bool allAbove = true;
    for (const auto& rule : result.Rules) {
        if (rule.Confidence < MIN_CONF) {
            allAbove = false;
            break;
        }
    }
    ASSERT_TRUE(allAbove,
                "All generated rules should have confidence >= minConfidence");
}

int main() {
    std::cout << "=== AprioriAlgorithm Unit Tests ===\n\n";

    TestGenerateCandidatesThreeItems();
    TestGenerateCandidatesSingleItem();
    TestGenerateCandidatesEmpty();
    TestCountOccurrences();
    TestCountOccurrencesNone();
    TestRunProducesRules();
    TestHighConfidenceReducesRules();
    TestRuleConfidenceAboveThreshold();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
