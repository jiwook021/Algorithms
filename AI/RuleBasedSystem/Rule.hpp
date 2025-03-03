/**
 * @file Rule.hpp
 * @brief Represents a single rule with condition and action
 */

#pragma once

#include "FactBase.hpp"

#include <functional>
#include <string>

/**
 * @brief Represents a rule in the knowledge base
 *
 * A rule has a condition (antecedent) and an action (consequent).
 * When the condition is met, the action is executed.
 */
class Rule {
public:
    /// Condition function type: takes a const reference to FactBase, returns bool
    using ConditionFunc = std::function<bool(const FactBase&)>;

    /// Action function type: takes a reference to FactBase, returns void
    using ActionFunc = std::function<void(FactBase&)>;

    /**
     * @brief Construct a rule with a name, condition, and action
     * @param name The name of the rule
     * @param condition The condition function
     * @param action The action function
     * @param priority The priority of the rule (higher values = higher priority)
     */
    Rule(std::string name, ConditionFunc condition, ActionFunc action, int priority = 0)
        : name_(std::move(name)),
          condition_(std::move(condition)),
          action_(std::move(action)),
          priority_(priority) {}

    /**
     * @brief Get the name of the rule
     * @return The rule name
     */
    [[nodiscard]] const std::string& GetName() const {
        return name_;
    }

    /**
     * @brief Get the priority of the rule
     * @return The rule priority
     */
    [[nodiscard]] int GetPriority() const {
        return priority_;
    }

    /**
     * @brief Evaluate the rule condition against the fact base
     * @param fact_base The fact base to evaluate against
     * @return true if the condition is met, false otherwise
     */
    [[nodiscard]] bool Evaluate(const FactBase& fact_base) const {
        if (!condition_) {
            return false;
        }
        return condition_(fact_base);
    }

    /**
     * @brief Execute the rule action on the fact base
     * @param fact_base The fact base to execute the action on
     */
    void Execute(FactBase& fact_base) const {
        if (action_) {
            action_(fact_base);
        }
    }

private:
    std::string name_;
    ConditionFunc condition_;
    ActionFunc action_;
    int priority_;
};
