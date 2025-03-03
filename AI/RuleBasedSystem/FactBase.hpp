/**
 * @file FactBase.hpp
 * @brief Thread-safe collection of facts (working memory)
 */

#pragma once

#include "Fact.hpp"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief A collection of facts (working memory)
 *
 * Thread-safe implementation for concurrent access.
 */
class FactBase {
public:
    /**
     * @brief Add or update a fact in the fact base
     * @param fact The fact to add or update
     * @return true if fact was added, false if it was updated
     */
    bool AddFact(const Fact& fact) {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto& name = fact.GetName();
        auto [it, inserted] = facts_.try_emplace(name, fact);

        if (!inserted) {
            it->second = fact;
        }

        return inserted;
    }

    /**
     * @brief Remove a fact from the fact base
     * @param fact_name The name of the fact to remove
     * @return true if the fact was removed, false if it wasn't found
     */
    bool RemoveFact(const std::string& fact_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        return facts_.erase(fact_name) > 0;
    }

    /**
     * @brief Check if a fact exists in the fact base
     * @param fact_name The name of the fact to check
     * @return true if the fact exists, false otherwise
     */
    [[nodiscard]] bool HasFact(const std::string& fact_name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return facts_.contains(fact_name);
    }

    /**
     * @brief Get a fact from the fact base
     * @param fact_name The name of the fact to get
     * @return Optional containing the fact if found, empty otherwise
     */
    [[nodiscard]] std::optional<Fact> GetFact(const std::string& fact_name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = facts_.find(fact_name);
        if (it != facts_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Get all facts in the fact base
     * @return Vector of all facts
     */
    [[nodiscard]] std::vector<Fact> GetAllFacts() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Fact> result;
        result.reserve(facts_.size());

        for (const auto& [_, fact] : facts_) {
            result.push_back(fact);
        }

        return result;
    }

    /**
     * @brief Clear all facts from the fact base
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        facts_.clear();
    }

    /**
     * @brief Get the number of facts in the fact base
     * @return Number of facts
     */
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return facts_.size();
    }

private:
    std::unordered_map<std::string, Fact> facts_;
    mutable std::mutex mutex_;  ///< Mutable to allow locking in const methods
};
