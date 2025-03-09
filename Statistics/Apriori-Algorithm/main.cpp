#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <algorithm>  // Required for std::find
#include <iomanip>    // For formatting output

// Function to generate candidate itemsets of size 2 from frequent single items
std::vector<std::pair<std::string, std::string>> generate_candidates(const std::set<std::string>& items) {
    std::vector<std::pair<std::string, std::string>> candidates;
    // Time Complexity: O(n²) where n is the number of items
    // Space Complexity: O(n²) for storing all possible pairs
    for (auto it1 = items.begin(); it1 != items.end(); ++it1) {
        auto it2 = it1;
        ++it2;
        for (; it2 != items.end(); ++it2) {
            candidates.push_back({*it1, *it2});
        }
    }
    return candidates;
}

// Function to count how many transactions contain a given itemset
int count_occurrences(const std::vector<std::vector<std::string>>& transactions, const std::set<std::string>& itemset) {
    int count = 0;
    // Time Complexity: O(t * i * m) where:
    // t is the number of transactions
    // i is the size of the itemset
    // m is the average number of items per transaction
    for (const auto& transaction : transactions) {
        bool all_present = true;
        for (const auto& item : itemset) {
            // Use std::find with proper iterator calls
            if (std::find(transaction.begin(), transaction.end(), item) == transaction.end()) {
                all_present = false;
                break;
            }
        }
        if (all_present) count++;
    }
    return count;
}

// Main Apriori Algorithm function
void apriori(const std::vector<std::vector<std::string>>& transactions, int min_support, double min_confidence) {
    // Step 1: Count occurrences of single items and find frequent ones
    // Time Complexity: O(t * m) where:
    // t is the number of transactions
    // m is the average number of items per transaction
    std::unordered_map<std::string, int> item_counts;
    for (const auto& transaction : transactions) {
        for (const auto& item : transaction) {
            item_counts[item]++;
        }
    }

    std::set<std::string> frequent_items;
    for (const auto& pair : item_counts) {
        if (pair.second >= min_support) {
            frequent_items.insert(pair.first);
        }
    }

    // Step 2: Generate candidate pairs from frequent items
    auto candidates = generate_candidates(frequent_items);

    // Step 3: Count occurrences of candidate pairs and find frequent ones
    std::unordered_map<std::string, int> pair_counts;
    for (const auto& candidate : candidates) {
        std::set<std::string> itemset = {candidate.first, candidate.second};
        int count = count_occurrences(transactions, itemset);
        if (count >= min_support) {
            pair_counts[candidate.first + "," + candidate.second] = count;
        }
    }

    // Step 4: Generate association rules and filter by confidence
    std::cout << "+-----------------+-----------------+------------------+\n";
    std::cout << "| Antecedent      | Consequent      | Confidence       |\n";
    std::cout << "+-----------------+-----------------+------------------+\n";
    
    for (const auto& pair : pair_counts) {
        std::string itemset_str = pair.first;
        int count = pair.second;

        size_t comma_pos = itemset_str.find(',');
        std::string item1 = itemset_str.substr(0, comma_pos);
        std::string item2 = itemset_str.substr(comma_pos + 1);

        // Rule: item1 -> item2
        int support_item1 = item_counts[item1];
        double confidence1 = static_cast<double>(count) / support_item1;
        if (confidence1 >= min_confidence) {
            std::cout << "| " << std::left << std::setw(15) << item1 
                      << " | " << std::setw(15) << item2 
                      << " | " << std::setw(16) << std::fixed << std::setprecision(4) << confidence1 << " |\n";
        }

        // Rule: item2 -> item1
        int support_item2 = item_counts[item2];
        double confidence2 = static_cast<double>(count) / support_item2;
        if (confidence2 >= min_confidence) {
            std::cout << "| " << std::left << std::setw(15) << item2 
                      << " | " << std::setw(15) << item1 
                      << " | " << std::setw(16) << std::fixed << std::setprecision(4) << confidence2 << " |\n";
        }
    }
    std::cout << "+-----------------+-----------------+------------------+\n";
    
    // Print some statistics
    std::cout << "\nStatistics:\n";
    std::cout << "Total transactions: " << transactions.size() << "\n";
    std::cout << "Unique items: " << item_counts.size() << "\n";
    std::cout << "Frequent items: " << frequent_items.size() << "\n";
    std::cout << "Frequent pairs: " << pair_counts.size() << "\n";
}

int main() {
    // Expanded dataset: 15 transactions with items
    std::vector<std::vector<std::string>> transactions = {
        // Original 5 transactions
        {"bread", "milk"},
        {"bread", "diapers", "beer", "eggs"},
        {"milk", "diapers", "beer", "cola"},
        {"bread", "milk", "diapers", "beer"},
        {"bread", "milk", "diapers", "cola"},
        
        // Added 10 more transactions
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

    int min_support = 3;        // Minimum number of occurrences for an itemset
    double min_confidence = 0.6; // Minimum confidence for a rule (60%)

    std::cout << "Association Rules (Apriori Algorithm):\n";
    apriori(transactions, min_support, min_confidence);

    return 0;
}