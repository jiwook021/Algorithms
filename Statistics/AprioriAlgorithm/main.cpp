/**
 * @file main.cpp
 * @brief Driver program for the Apriori association rule mining algorithm.
 */

#include <iomanip>
#include <iostream>

#include "AprioriAlgorithm.hpp"

/**
 * @brief Print the association rules and statistics in tabular form.
 */
void PrintResult(const AprioriResult& result, std::size_t transactionCount) {
    std::cout << "+-----------------+-----------------+------------------+\n";
    std::cout << "| Antecedent      | Consequent      | Confidence       |\n";
    std::cout << "+-----------------+-----------------+------------------+\n";

    for (const auto& rule : result.Rules) {
        std::cout << "| " << std::left << std::setw(15) << rule.Antecedent
                  << " | " << std::setw(15) << rule.Consequent
                  << " | " << std::setw(16) << std::fixed
                  << std::setprecision(4) << rule.Confidence << " |\n";
    }
    std::cout << "+-----------------+-----------------+------------------+\n";

    std::cout << "\nStatistics:\n";
    std::cout << "Total transactions: " << transactionCount << "\n";
    std::cout << "Unique items: " << result.ItemCounts.size() << "\n";
    std::cout << "Frequent items: " << result.FrequentItems.size() << "\n";
    std::cout << "Frequent pairs: " << result.PairCounts.size() << "\n";
}

int main() {
    AprioriAlgorithm::Transactions transactions = {
        {"bread", "milk"},
        {"bread", "diapers", "beer", "eggs"},
        {"milk", "diapers", "beer", "cola"},
        {"bread", "milk", "diapers", "beer"},
        {"bread", "milk", "diapers", "cola"},
        {"bread", "eggs", "milk"},
        {"diapers", "beer", "cola"},
        {"bread", "milk", "eggs"},
        {"bread", "diapers", "cola"},
        {"milk", "diapers", "eggs"},
        {"bread", "beer", "cola"},
        {"diapers", "milk", "beer"},
        {"bread", "milk", "cola"},
        {"bread", "diapers", "beer"},
        {"milk", "eggs", "cola"}
    };

    const int MIN_SUPPORT = 3;
    const double MIN_CONFIDENCE = 0.6;

    AprioriAlgorithm algorithm;
    const AprioriResult result =
        algorithm.Run(transactions, MIN_SUPPORT, MIN_CONFIDENCE);

    std::cout << "Association Rules (Apriori Algorithm):\n";
    PrintResult(result, transactions.size());

    return 0;
}
