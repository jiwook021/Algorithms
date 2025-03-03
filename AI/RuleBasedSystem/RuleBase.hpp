/**
 * @file RuleBase.hpp
 * @brief Thread-safe collection of rules
 */

#pragma once

#include "Rule.hpp"

#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief A collection of rules
 *
 * Thread-safe implementation for concurrent access.
 */
class RuleBase {
public:
    /**
     * @brief Add a rule to the rule base
     * @param rule The rule to add
     * @return true if the rule was added, false if a rule with the same name already exists
     */
    bool AddRule(const Rule& rule) {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto& name = rule.GetName();
        auto [it, inserted] = rules_.try_emplace(name, rule);
        return inserted;
    }

    /**
     * @brief Remove a rule from the rule base
     * @param rule_name The name of the rule to remove
     * @return true if the rule was removed, false if it wasn't found
     */
    bool RemoveRule(const std::string& rule_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        return rules_.erase(rule_name) > 0;
    }

    /**
     * @brief Get a rule from the rule base
     * @param rule_name The name of the rule to get
     * @return Optional containing the rule if found, empty otherwise
     */
    [[nodiscard]] std::optional<Rule> GetRule(const std::string& rule_name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rules_.find(rule_name);
        if (it != rules_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Get all rules in the rule base, sorted by priority (highest first)
     * @return Vector of all rules
     */
    [[nodiscard]] std::vector<Rule> GetAllRules() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Rule> result;
        result.reserve(rules_.size());

        for (const auto& [_, rule] : rules_) {
            result.push_back(rule);
        }

        // Sort rules by priority (highest first)
        std::sort(result.begin(), result.end(),
                  [](const Rule& a, const Rule& b) {
                      return a.GetPriority() > b.GetPriority();
                  });

        return result;
    }

    /**
     * @brief Clear all rules from the rule base
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        rules_.clear();
    }

    /**
     * @brief Get the number of rules in the rule base
     * @return Number of rules
     */
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rules_.size();
    }

private:
    std::unordered_map<std::string, Rule> rules_;
    mutable std::mutex mutex_;  ///< Mutable to allow locking in const methods
};
