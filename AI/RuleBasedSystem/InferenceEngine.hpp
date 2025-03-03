/**
 * @file InferenceEngine.hpp
 * @brief Forward-chaining inference engine
 * @details Iteratively evaluates rules against the fact base until a fixpoint
 *          (no new facts derived) or the iteration cap is reached.
 *
 * Time Complexity:
 *   - Rule evaluation per cycle: O(R * F) where R = rules, F = facts
 *   - Worst-case total: O(I * R * F) where I = max iterations
 *
 * Space Complexity: O(R + F)
 */

#pragma once

#include "FactBase.hpp"
#include "RuleBase.hpp"

#include <memory>
#include <stdexcept>
#include <unordered_set>

/**
 * @brief The inference engine that applies rules to facts
 *
 * Implements a forward-chaining algorithm.
 */
class InferenceEngine {
public:
    /**
     * @brief Construct an inference engine with a rule base and fact base
     * @param rule_base The rule base to use
     * @param fact_base The fact base to use
     * @param max_iterations Maximum number of iterations to prevent infinite loops
     */
    InferenceEngine(std::shared_ptr<RuleBase> rule_base,
                    std::shared_ptr<FactBase> fact_base,
                    size_t max_iterations = 100)
        : rule_base_(std::move(rule_base)),
          fact_base_(std::move(fact_base)),
          max_iterations_(max_iterations) {
        if (!rule_base_) {
            throw std::invalid_argument("Rule base cannot be null");
        }
        if (!fact_base_) {
            throw std::invalid_argument("Fact base cannot be null");
        }
    }

    /**
     * @brief Run the inference engine until fixpoint or max iterations
     * @return Number of rules executed
     */
    size_t Run() {
        size_t iteration_count = 0;
        size_t rule_execution_count = 0;
        bool changes_applied;

        std::unordered_set<std::string> fired_rules;

        do {
            changes_applied = false;
            ++iteration_count;

            const auto rules = rule_base_->GetAllRules();

            for (const auto& rule : rules) {
                if (fired_rules.contains(rule.GetName())) {
                    continue;
                }

                if (rule.Evaluate(*fact_base_)) {
                    auto before_size = fact_base_->size();
                    auto before_facts = fact_base_->GetAllFacts();

                    rule.Execute(*fact_base_);
                    ++rule_execution_count;

                    auto after_size = fact_base_->size();
                    if (after_size != before_size) {
                        changes_applied = true;
                    } else {
                        auto after_facts = fact_base_->GetAllFacts();
                        for (size_t i = 0; i < before_facts.size(); ++i) {
                            if (before_facts[i].ToString() != after_facts[i].ToString()) {
                                changes_applied = true;
                                break;
                            }
                        }
                    }

                    fired_rules.insert(rule.GetName());
                }
            }
        } while (changes_applied && iteration_count < max_iterations_);

        return rule_execution_count;
    }

    /**
     * @brief Set the maximum number of iterations
     * @param max_iterations The maximum number of iterations
     */
    void SetMaxIterations(size_t max_iterations) {
        max_iterations_ = max_iterations;
    }

    /**
     * @brief Get the maximum number of iterations
     * @return The maximum number of iterations
     */
    [[nodiscard]] size_t GetMaxIterations() const {
        return max_iterations_;
    }

private:
    std::shared_ptr<RuleBase> rule_base_;
    std::shared_ptr<FactBase> fact_base_;
    size_t max_iterations_;
};
