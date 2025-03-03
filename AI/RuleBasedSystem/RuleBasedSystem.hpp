/**
 * @file RuleBasedSystem.hpp
 * @brief Top-level include-all header and facade for the rule-based system
 * @details Includes Fact, FactBase, Rule, RuleBase, InferenceEngine and
 *          provides the RuleBasedSystem facade that wires them together.
 *
 * Include only this header to get the complete rule-based system.
 *
 * Time Complexity:
 *   - Rule evaluation: O(R * F) where R = rules, F = facts
 *   - Fact insertion: O(1) amortized
 *
 * Space Complexity: O(R + F)
 */

#pragma once

// ---- component headers (order matters: each may depend on the previous) ----
#include "Fact.hpp"
#include "FactBase.hpp"
#include "Rule.hpp"
#include "RuleBase.hpp"
#include "InferenceEngine.hpp"

#include <iostream>
#include <memory>
#include <string>

/**
 * @brief A rule-based system facade that encapsulates all components
 */
class RuleBasedSystem {
public:
    /**
     * @brief Construct a rule-based system
     * @param name The name of the system
     */
    explicit RuleBasedSystem(std::string name)
        : name_(std::move(name)),
          rule_base_(std::make_shared<RuleBase>()),
          fact_base_(std::make_shared<FactBase>()),
          inference_engine_(std::make_shared<InferenceEngine>(rule_base_, fact_base_)) {}

    /**
     * @brief Get the name of the system
     * @return The system name
     */
    [[nodiscard]] const std::string& GetName() const {
        return name_;
    }

    /**
     * @brief Add a fact to the system
     * @param fact The fact to add
     * @return true if the fact was added, false if it was updated
     */
    bool AddFact(const Fact& fact) {
        return fact_base_->AddFact(fact);
    }

    /**
     * @brief Add a rule to the system
     * @param rule The rule to add
     * @return true if the rule was added, false if a rule with the same name already exists
     */
    bool AddRule(const Rule& rule) {
        return rule_base_->AddRule(rule);
    }

    /**
     * @brief Run the inference engine
     * @return Number of rules executed
     */
    size_t Run() {
        return inference_engine_->Run();
    }

    /**
     * @brief Get the fact base
     * @return Shared pointer to the fact base
     */
    [[nodiscard]] std::shared_ptr<FactBase> GetFactBase() const {
        return fact_base_;
    }

    /**
     * @brief Get the rule base
     * @return Shared pointer to the rule base
     */
    [[nodiscard]] std::shared_ptr<RuleBase> GetRuleBase() const {
        return rule_base_;
    }

    /**
     * @brief Get the inference engine
     * @return Shared pointer to the inference engine
     */
    [[nodiscard]] std::shared_ptr<InferenceEngine> GetInferenceEngine() const {
        return inference_engine_;
    }

private:
    std::string name_;
    std::shared_ptr<RuleBase> rule_base_;
    std::shared_ptr<FactBase> fact_base_;
    std::shared_ptr<InferenceEngine> inference_engine_;
};

/**
 * @brief Demo functions for the rule-based system (called from main)
 */
namespace Demo {

/**
 * @brief Test the rule-based system with a simple weather example
 */
inline void RunSimpleTest() {
    std::cout << "=== Simple Rule-Based System Test ===" << std::endl;

    RuleBasedSystem system("SimpleTest");

    system.AddFact(Fact("temperature", 75.0));
    system.AddFact(Fact("humidity", 65.0));
    system.AddFact(Fact("is_raining", false));

    // Rule 1: high temp + humidity -> uncomfortable
    system.AddRule(Rule(
        "high_temp_rule",
        [](const FactBase& facts) {
            auto temp = facts.GetFact("temperature");
            auto hum  = facts.GetFact("humidity");
            if (!temp || !hum) return false;
            return std::get<double>(temp->GetValue()) > 80.0
                && std::get<double>(hum->GetValue()) > 60.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("comfort", std::string("uncomfortable")));
            std::cout << "Rule 'high_temp_rule' executed: uncomfortable.\n";
        }));

    // Rule 2: rain -> wet ground
    system.AddRule(Rule(
        "rain_rule",
        [](const FactBase& facts) {
            auto r = facts.GetFact("is_raining");
            return r && std::get<bool>(r->GetValue());
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("ground_condition", std::string("wet")));
            std::cout << "Rule 'rain_rule' executed: ground is wet.\n";
        }));

    // Rule 3: cold temp -> cold comfort
    system.AddRule(Rule(
        "cold_temp_rule",
        [](const FactBase& facts) {
            auto t = facts.GetFact("temperature");
            return t && std::get<double>(t->GetValue()) < 60.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("comfort", std::string("cold")));
            std::cout << "Rule 'cold_temp_rule' executed: cold.\n";
        }));

    std::cout << "Initial facts:\n";
    for (const auto& f : system.GetFactBase()->GetAllFacts())
        std::cout << "  " << f.ToString() << "\n";

    std::cout << "\nRunning inference engine...\n";
    size_t fired = system.Run();
    std::cout << "Rules executed: " << fired << "\n";

    std::cout << "\nFinal facts:\n";
    for (const auto& f : system.GetFactBase()->GetAllFacts())
        std::cout << "  " << f.ToString() << "\n";

    std::cout << "\nUpdating temperature to 85.0...\n";
    system.AddFact(Fact("temperature", 85.0));

    std::cout << "Running inference engine again...\n";
    fired = system.Run();
    std::cout << "Rules executed: " << fired << "\n";

    std::cout << "\nFinal facts after update:\n";
    for (const auto& f : system.GetFactBase()->GetAllFacts())
        std::cout << "  " << f.ToString() << "\n";
}

/**
 * @brief Medical-diagnosis demo
 */
inline void RunMedicalDiagnosisTest() {
    std::cout << "\n=== Medical Diagnosis Rule-Based System Test ===" << std::endl;

    RuleBasedSystem diag("MedicalDiagnosis");

    diag.AddFact(Fact("has_fever", true));
    diag.AddFact(Fact("body_temperature", 101.5));
    diag.AddFact(Fact("has_cough", true));
    diag.AddFact(Fact("has_rash", false));
    diag.AddFact(Fact("has_joint_pain", false));
    diag.AddFact(Fact("has_sore_throat", true));

    // Common Cold (priority 10)
    diag.AddRule(Rule(
        "common_cold_rule",
        [](const FactBase& facts) {
            auto fever = facts.GetFact("has_fever");
            auto cough = facts.GetFact("has_cough");
            auto throat = facts.GetFact("has_sore_throat");
            if (!fever || !cough || !throat) return false;
            auto bt = facts.GetFact("body_temperature");
            double temp = bt ? std::get<double>(bt->GetValue()) : 98.6;
            return std::get<bool>(fever->GetValue())
                && std::get<bool>(cough->GetValue())
                && std::get<bool>(throat->GetValue())
                && temp < 102.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("diagnosis", std::string("Common Cold")));
            facts.AddFact(Fact("treatment", std::string("Rest, fluids, OTC medication")));
            std::cout << "Diagnosed: Common Cold\n";
        },
        10));

    // Flu (priority 20)
    diag.AddRule(Rule(
        "flu_rule",
        [](const FactBase& facts) {
            auto fever = facts.GetFact("has_fever");
            auto cough = facts.GetFact("has_cough");
            if (!fever || !cough) return false;
            auto bt = facts.GetFact("body_temperature");
            double temp = bt ? std::get<double>(bt->GetValue()) : 98.6;
            return std::get<bool>(fever->GetValue())
                && std::get<bool>(cough->GetValue())
                && temp >= 102.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("diagnosis", std::string("Influenza (Flu)")));
            facts.AddFact(Fact("treatment", std::string("Rest, fluids, antivirals")));
            std::cout << "Diagnosed: Influenza\n";
        },
        20));

    // Allergic Reaction (priority 30)
    diag.AddRule(Rule(
        "allergy_rule",
        [](const FactBase& facts) {
            auto rash = facts.GetFact("has_rash");
            return rash && std::get<bool>(rash->GetValue());
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("diagnosis", std::string("Allergic Reaction")));
            facts.AddFact(Fact("treatment", std::string("Antihistamines")));
            std::cout << "Diagnosed: Allergic Reaction\n";
        },
        30));

    std::cout << "Patient symptoms:\n";
    for (const auto& f : diag.GetFactBase()->GetAllFacts())
        std::cout << "  " << f.ToString() << "\n";

    std::cout << "\nRunning diagnosis...\n";
    size_t fired = diag.Run();
    std::cout << "Rules executed: " << fired << "\n";

    auto dx = diag.GetFactBase()->GetFact("diagnosis");
    auto tx = diag.GetFactBase()->GetFact("treatment");
    std::cout << "\nDiagnosis: "
              << (dx ? std::get<std::string>(dx->GetValue()) : "none") << "\n";
    std::cout << "Treatment: "
              << (tx ? std::get<std::string>(tx->GetValue()) : "none") << "\n";

    std::cout << "\nUpdating: temp->103, joint pain->true...\n";
    diag.AddFact(Fact("body_temperature", 103.0));
    diag.AddFact(Fact("has_joint_pain", true));
    diag.GetFactBase()->RemoveFact("diagnosis");
    diag.GetFactBase()->RemoveFact("treatment");

    std::cout << "Running diagnosis again...\n";
    fired = diag.Run();
    std::cout << "Rules executed: " << fired << "\n";

    dx = diag.GetFactBase()->GetFact("diagnosis");
    tx = diag.GetFactBase()->GetFact("treatment");
    std::cout << "\nUpdated Diagnosis: "
              << (dx ? std::get<std::string>(dx->GetValue()) : "none") << "\n";
    std::cout << "Updated Treatment: "
              << (tx ? std::get<std::string>(tx->GetValue()) : "none") << "\n";
}

}  // namespace Demo
