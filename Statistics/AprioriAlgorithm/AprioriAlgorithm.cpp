/**
 * @file AprioriAlgorithm.cpp
 * @brief Implementation of the Apriori algorithm for association rule mining.
 */

#include "AprioriAlgorithm.hpp"

std::vector<std::pair<std::string, std::string>>
AprioriAlgorithm::GenerateCandidates(const std::set<std::string>& items) const {
    std::vector<std::pair<std::string, std::string>> candidates;
    for (auto it1 = items.begin(); it1 != items.end(); ++it1) {
        auto it2 = it1;
        ++it2;
        for (; it2 != items.end(); ++it2) {
            candidates.push_back({*it1, *it2});
        }
    }
    return candidates;
}

int AprioriAlgorithm::CountOccurrences(const Transactions& transactionsData,
                                        const std::set<std::string>& itemset) const {
    int count = 0;
    for (const auto& transaction : transactionsData) {
        bool allPresent = true;
        for (const auto& item : itemset) {
            if (std::find(transaction.begin(), transaction.end(), item) ==
                transaction.end()) {
                allPresent = false;
                break;
            }
        }
        if (allPresent) {
            ++count;
        }
    }
    return count;
}

AprioriResult AprioriAlgorithm::Run(const Transactions& transactionsData,
                                     int minSupport,
                                     double minConfidence) const {
    AprioriResult result;

    // Phase 1: Count individual item support
    for (const auto& transaction : transactionsData) {
        for (const auto& item : transaction) {
            result.ItemCounts[item]++;
        }
    }

    // Filter frequent single items
    for (const auto& pair : result.ItemCounts) {
        if (pair.second >= minSupport) {
            result.FrequentItems.insert(pair.first);
        }
    }

    // Phase 2: Count frequent pairs
    auto candidates = GenerateCandidates(result.FrequentItems);
    for (const auto& candidate : candidates) {
        const std::set<std::string> itemset = {candidate.first, candidate.second};
        const int count = CountOccurrences(transactionsData, itemset);
        if (count >= minSupport) {
            result.PairCounts[candidate.first + "," + candidate.second] = count;
        }
    }

    // Phase 3: Generate association rules from frequent pairs
    for (const auto& pair : result.PairCounts) {
        const std::string& itemsetString = pair.first;
        const int count = pair.second;

        const size_t commaPos = itemsetString.find(',');
        const std::string item1 = itemsetString.substr(0, commaPos);
        const std::string item2 = itemsetString.substr(commaPos + 1);

        const int supportItem1 = result.ItemCounts[item1];
        const double confidence1 = static_cast<double>(count) / supportItem1;
        if (confidence1 >= minConfidence) {
            result.Rules.push_back({item1, item2, confidence1, count});
        }

        const int supportItem2 = result.ItemCounts[item2];
        const double confidence2 = static_cast<double>(count) / supportItem2;
        if (confidence2 >= minConfidence) {
            result.Rules.push_back({item2, item1, confidence2, count});
        }
    }

    return result;
}
