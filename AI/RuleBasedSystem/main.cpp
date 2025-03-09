#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <any>
#include <algorithm>
#include <stdexcept>
#include <sstream>  // Instead of format
#include <atomic>
#include <thread>

/**
 * @brief A simple rule-based system implementation
 * 
 * This implementation follows a forward-chaining inference engine pattern
 * where rules are evaluated against facts to derive new facts.
 * 
 * Time Complexity: 
 * - Rule evaluation: O(R * F) where R is number of rules and F is number of facts
 * - Fact insertion: O(1) amortized
 * 
 * Space Complexity: O(R + F) where R is number of rules and F is number of facts
 */

// Forward declarations
class Fact;
class FactBase;
class Rule;
class RuleBase;
class InferenceEngine;

/**
 * @brief Represents a fact in the knowledge base
 * 
 * Facts can be of different types, using std::variant to support multiple types.
 */
class Fact {
public:
    // Supported fact types: boolean, integer, double, and string
    using FactValue = std::variant<bool, int, double, std::string>;
    
    /**
     * @brief Construct a fact with a name and value
     * @param name The name of the fact
     * @param value The value of the fact
     */
    Fact(std::string name, FactValue value)
        : name_(std::move(name)), value_(std::move(value)) {}
    
    /**
     * @brief Get the name of the fact
     * @return The fact name
     */
    [[nodiscard]] const std::string& getName() const {
        return name_;
    }
    
    /**
     * @brief Get the value of the fact
     * @return The fact value
     */
    [[nodiscard]] const FactValue& getValue() const {
        return value_;
    }
    
    /**
     * @brief Update the fact value
     * @param value The new value
     */
    void setValue(FactValue value) {
        value_ = std::move(value);
    }
    
    /**
     * @brief Convert fact to string representation
     * @return String representation of the fact
     */
    [[nodiscard]] std::string toString() const {
        std::string result = name_ + " = ";
        
        std::visit([&result](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::string>) {
                result += "\"" + val + "\"";
            } else if constexpr (std::is_same_v<T, bool>) {
                result += val ? "true" : "false";
            } else {
                result += std::to_string(val);
            }
        }, value_);
        
        return result;
    }

private:
    std::string name_;
    FactValue value_;
};

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
    bool addFact(const Fact& fact) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed here to prevent concurrent modification 
        // of the facts map, which could lead to data races
        
        const auto& name = fact.getName();
        auto [it, inserted] = facts_.try_emplace(name, fact);
        
        if (!inserted) {
            it->second = fact;
        }
        
        return inserted;
    }
    
    /**
     * @brief Remove a fact from the fact base
     * @param factName The name of the fact to remove
     * @return true if the fact was removed, false if it wasn't found
     */
    bool removeFact(const std::string& factName) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed to prevent concurrent modification
        
        return facts_.erase(factName) > 0;
    }
    
    /**
     * @brief Check if a fact exists in the fact base
     * @param factName The name of the fact to check
     * @return true if the fact exists, false otherwise
     */
    [[nodiscard]] bool hasFact(const std::string& factName) const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the facts map
        
        return facts_.contains(factName);
    }
    
    /**
     * @brief Get a fact from the fact base
     * @param factName The name of the fact to get
     * @return Optional containing the fact if found, empty otherwise
     */
    [[nodiscard]] std::optional<Fact> getFact(const std::string& factName) const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the facts map
        
        auto it = facts_.find(factName);
        if (it != facts_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get all facts in the fact base
     * @return Vector of all facts
     */
    [[nodiscard]] std::vector<Fact> getAllFacts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the facts map
        
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
        // Lock is needed to prevent concurrent modification
        
        facts_.clear();
    }
    
    /**
     * @brief Get the number of facts in the fact base
     * @return Number of facts
     */
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the facts map
        
        return facts_.size();
    }

private:
    std::unordered_map<std::string, Fact> facts_;
    mutable std::mutex mutex_; // Mutable to allow locking in const methods
};

/**
 * @brief Represents a rule in the knowledge base
 * 
 * A rule has a condition (antecedent) and an action (consequent).
 * When the condition is met, the action is executed.
 */
class Rule {
public:
    // Condition function type: takes a const reference to FactBase, returns bool
    using ConditionFunc = std::function<bool(const FactBase&)>;
    
    // Action function type: takes a reference to FactBase, returns void
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
    [[nodiscard]] const std::string& getName() const {
        return name_;
    }
    
    /**
     * @brief Get the priority of the rule
     * @return The rule priority
     */
    [[nodiscard]] int getPriority() const {
        return priority_;
    }
    
    /**
     * @brief Evaluate the rule condition against the fact base
     * @param factBase The fact base to evaluate against
     * @return true if the condition is met, false otherwise
     */
    [[nodiscard]] bool evaluate(const FactBase& factBase) const {
        if (!condition_) {
            return false;
        }
        return condition_(factBase);
    }
    
    /**
     * @brief Execute the rule action on the fact base
     * @param factBase The fact base to execute the action on
     */
    void execute(FactBase& factBase) const {
        if (action_) {
            action_(factBase);
        }
    }

private:
    std::string name_;
    ConditionFunc condition_;
    ActionFunc action_;
    int priority_;
};

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
    bool addRule(const Rule& rule) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed to prevent concurrent modification
        
        const auto& name = rule.getName();
        auto [it, inserted] = rules_.try_emplace(name, rule);
        return inserted;
    }
    
    /**
     * @brief Remove a rule from the rule base
     * @param ruleName The name of the rule to remove
     * @return true if the rule was removed, false if it wasn't found
     */
    bool removeRule(const std::string& ruleName) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed to prevent concurrent modification
        
        return rules_.erase(ruleName) > 0;
    }
    
    /**
     * @brief Get a rule from the rule base
     * @param ruleName The name of the rule to get
     * @return Optional containing the rule if found, empty otherwise
     */
    [[nodiscard]] std::optional<Rule> getRule(const std::string& ruleName) const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the rules map
        
        auto it = rules_.find(ruleName);
        if (it != rules_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get all rules in the rule base, sorted by priority
     * @return Vector of all rules
     */
    [[nodiscard]] std::vector<Rule> getAllRules() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the rules map
        
        std::vector<Rule> result;
        result.reserve(rules_.size());
        
        for (const auto& [_, rule] : rules_) {
            result.push_back(rule);
        }
        
        // Sort rules by priority (highest first)
        std::sort(result.begin(), result.end(), 
                 [](const Rule& a, const Rule& b) {
                     return a.getPriority() > b.getPriority();
                 });
        
        return result;
    }
    
    /**
     * @brief Clear all rules from the rule base
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed to prevent concurrent modification
        
        rules_.clear();
    }
    
    /**
     * @brief Get the number of rules in the rule base
     * @return Number of rules
     */
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Lock is needed for consistent reading of the rules map
        
        return rules_.size();
    }

private:
    std::unordered_map<std::string, Rule> rules_;
    mutable std::mutex mutex_; // Mutable to allow locking in const methods
};

/**
 * @brief The inference engine that applies rules to facts
 * 
 * Implements a forward-chaining algorithm.
 */
class InferenceEngine {
public:
    /**
     * @brief Construct an inference engine with a rule base and fact base
     * @param ruleBase The rule base to use
     * @param factBase The fact base to use
     * @param maxIterations Maximum number of iterations to prevent infinite loops
     */
    InferenceEngine(std::shared_ptr<RuleBase> ruleBase, 
                   std::shared_ptr<FactBase> factBase,
                   size_t maxIterations = 100)
        : ruleBase_(std::move(ruleBase)), 
          factBase_(std::move(factBase)),
          maxIterations_(maxIterations) {
        
        // Validate inputs
        if (!ruleBase_) {
            throw std::invalid_argument("Rule base cannot be null");
        }
        if (!factBase_) {
            throw std::invalid_argument("Fact base cannot be null");
        }
    }
    
    /**
     * @brief Run the inference engine until no more rules can be applied or max iterations reached
     * @return Number of rules executed
     */
    size_t run() {
        size_t iterationCount = 0;
        size_t ruleExecutionCount = 0;
        bool changesApplied;
        
        std::unordered_set<std::string> firedRules; // Track fired rules for this run
        
        do {
            changesApplied = false;
            iterationCount++;
            
            // Get all rules, sorted by priority
            const auto rules = ruleBase_->getAllRules();
            
            for (const auto& rule : rules) {
                // Skip rules already fired in this run
                if (firedRules.contains(rule.getName())) {
                    continue;
                }
                
                // Evaluate rule condition
                if (rule.evaluate(*factBase_)) {
                    // Create a snapshot of the fact base for comparison
                    auto beforeSize = factBase_->size();
                    auto beforeFacts = factBase_->getAllFacts();
                    
                    // Execute rule action
                    rule.execute(*factBase_);
                    ruleExecutionCount++;
                    
                    // Check if any changes were made
                    auto afterSize = factBase_->size();
                    if (afterSize != beforeSize) {
                        changesApplied = true;
                    } else {
                        auto afterFacts = factBase_->getAllFacts();
                        for (size_t i = 0; i < beforeFacts.size(); ++i) {
                            if (beforeFacts[i].toString() != afterFacts[i].toString()) {
                                changesApplied = true;
                                break;
                            }
                        }
                    }
                    
                    // Add rule to fired rules
                    firedRules.insert(rule.getName());
                }
            }
            
        } while (changesApplied && iterationCount < maxIterations_);
        
        return ruleExecutionCount;
    }
    
    /**
     * @brief Set the maximum number of iterations
     * @param maxIterations The maximum number of iterations
     */
    void setMaxIterations(size_t maxIterations) {
        maxIterations_ = maxIterations;
    }
    
    /**
     * @brief Get the maximum number of iterations
     * @return The maximum number of iterations
     */
    [[nodiscard]] size_t getMaxIterations() const {
        return maxIterations_;
    }

private:
    std::shared_ptr<RuleBase> ruleBase_;
    std::shared_ptr<FactBase> factBase_;
    size_t maxIterations_; // To prevent infinite loops
};

/**
 * @brief A simple rule-based system that encapsulates all components
 */
class RuleBasedSystem {
public:
    /**
     * @brief Construct a rule-based system
     * @param name The name of the system
     */
    explicit RuleBasedSystem(std::string name)
        : name_(std::move(name)),
          ruleBase_(std::make_shared<RuleBase>()),
          factBase_(std::make_shared<FactBase>()),
          inferenceEngine_(std::make_shared<InferenceEngine>(ruleBase_, factBase_)) {}
    
    /**
     * @brief Get the name of the system
     * @return The system name
     */
    [[nodiscard]] const std::string& getName() const {
        return name_;
    }
    
    /**
     * @brief Add a fact to the system
     * @param fact The fact to add
     * @return true if the fact was added, false if it was updated
     */
    bool addFact(const Fact& fact) {
        return factBase_->addFact(fact);
    }
    
    /**
     * @brief Add a rule to the system
     * @param rule The rule to add
     * @return true if the rule was added, false if a rule with the same name already exists
     */
    bool addRule(const Rule& rule) {
        return ruleBase_->addRule(rule);
    }
    
    /**
     * @brief Run the inference engine
     * @return Number of rules executed
     */
    size_t run() {
        return inferenceEngine_->run();
    }
    
    /**
     * @brief Get the fact base
     * @return Shared pointer to the fact base
     */
    [[nodiscard]] std::shared_ptr<FactBase> getFactBase() const {
        return factBase_;
    }
    
    /**
     * @brief Get the rule base
     * @return Shared pointer to the rule base
     */
    [[nodiscard]] std::shared_ptr<RuleBase> getRuleBase() const {
        return ruleBase_;
    }
    
    /**
     * @brief Get the inference engine
     * @return Shared pointer to the inference engine
     */
    [[nodiscard]] std::shared_ptr<InferenceEngine> getInferenceEngine() const {
        return inferenceEngine_;
    }

private:
    std::string name_;
    std::shared_ptr<RuleBase> ruleBase_;
    std::shared_ptr<FactBase> factBase_;
    std::shared_ptr<InferenceEngine> inferenceEngine_;
};

// Test functions for the rule-based system
namespace test {
    /**
     * @brief Test the rule-based system with a simple example
     */
    void runSimpleTest() {
        std::cout << "=== Simple Rule-Based System Test ===" << std::endl;
        
        // Create a rule-based system
        RuleBasedSystem system("SimpleTest");
        
        // Add initial facts
        system.addFact(Fact("temperature", 75.0));
        system.addFact(Fact("humidity", 65.0));
        system.addFact(Fact("is_raining", false));
        
        // Add rules
        // Rule 1: If temperature > 80 and humidity > 60, then it's uncomfortable
        system.addRule(Rule(
            "high_temp_rule",
            [](const FactBase& facts) {
                auto tempFact = facts.getFact("temperature");
                auto humidityFact = facts.getFact("humidity");
                
                if (!tempFact || !humidityFact) {
                    return false;
                }
                
                double temp = std::get<double>(tempFact->getValue());
                double humidity = std::get<double>(humidityFact->getValue());
                
                return temp > 80.0 && humidity > 60.0;
            },
            [](FactBase& facts) {
                facts.addFact(Fact("comfort", std::string("uncomfortable")));
                std::cout << "Rule 'high_temp_rule' executed: It's uncomfortable due to high temperature and humidity." << std::endl;
            }
        ));
        
        // Rule 2: If it's raining, then it's wet outside
        system.addRule(Rule(
            "rain_rule",
            [](const FactBase& facts) {
                auto isRainingFact = facts.getFact("is_raining");
                
                if (!isRainingFact) {
                    return false;
                }
                
                return std::get<bool>(isRainingFact->getValue());
            },
            [](FactBase& facts) {
                facts.addFact(Fact("ground_condition", std::string("wet")));
                std::cout << "Rule 'rain_rule' executed: Ground is wet due to rain." << std::endl;
            }
        ));
        
        // Rule 3: If temperature < 60, then it's cold
        system.addRule(Rule(
            "cold_temp_rule",
            [](const FactBase& facts) {
                auto tempFact = facts.getFact("temperature");
                
                if (!tempFact) {
                    return false;
                }
                
                double temp = std::get<double>(tempFact->getValue());
                
                return temp < 60.0;
            },
            [](FactBase& facts) {
                facts.addFact(Fact("comfort", std::string("cold")));
                std::cout << "Rule 'cold_temp_rule' executed: It's cold due to low temperature." << std::endl;
            }
        ));
        
        // Print initial facts
        std::cout << "Initial facts:" << std::endl;
        for (const auto& fact : system.getFactBase()->getAllFacts()) {
            std::cout << "  " << fact.toString() << std::endl;
        }
        
        // Run the inference engine
        std::cout << "\nRunning inference engine..." << std::endl;
        size_t rulesExecuted = system.run();
        std::cout << "Number of rules executed: " << rulesExecuted << std::endl;
        
        // Print final facts
        std::cout << "\nFinal facts:" << std::endl;
        for (const auto& fact : system.getFactBase()->getAllFacts()) {
            std::cout << "  " << fact.toString() << std::endl;
        }
        
        // Update a fact and run again
        std::cout << "\nUpdating temperature to 85.0..." << std::endl;
        system.addFact(Fact("temperature", 85.0));
        
        // Run the inference engine again
        std::cout << "Running inference engine again..." << std::endl;
        rulesExecuted = system.run();
        std::cout << "Number of rules executed: " << rulesExecuted << std::endl;
        
        // Print final facts
        std::cout << "\nFinal facts after update:" << std::endl;
        for (const auto& fact : system.getFactBase()->getAllFacts()) {
            std::cout << "  " << fact.toString() << std::endl;
        }
    }
    
    /**
     * @brief Test the rule-based system with a more complex example: a medical diagnosis system
     */
    void runMedicalDiagnosisTest() {
        std::cout << "\n=== Medical Diagnosis Rule-Based System Test ===" << std::endl;
        
        // Create a rule-based system for medical diagnosis
        RuleBasedSystem diagnosisSystem("MedicalDiagnosis");
        
        // Add initial facts (symptoms)
        diagnosisSystem.addFact(Fact("has_fever", true));
        diagnosisSystem.addFact(Fact("body_temperature", 101.5));
        diagnosisSystem.addFact(Fact("has_cough", true));
        diagnosisSystem.addFact(Fact("has_rash", false));
        diagnosisSystem.addFact(Fact("has_joint_pain", false));
        diagnosisSystem.addFact(Fact("has_sore_throat", true));
        
        // Add rules for diagnosing conditions
        
        // Rule 1: Common Cold
        diagnosisSystem.addRule(Rule(
            "common_cold_rule",
            [](const FactBase& facts) {
                auto hasFever = facts.getFact("has_fever");
                auto hasCough = facts.getFact("has_cough");
                auto hasSoreThroat = facts.getFact("has_sore_throat");
                
                if (!hasFever || !hasCough || !hasSoreThroat) {
                    return false;
                }
                
                bool fever = std::get<bool>(hasFever->getValue());
                bool cough = std::get<bool>(hasCough->getValue());
                bool soreThroat = std::get<bool>(hasSoreThroat->getValue());
                
                auto bodyTemp = facts.getFact("body_temperature");
                double temp = bodyTemp ? std::get<double>(bodyTemp->getValue()) : 98.6;
                
                // Common cold: mild fever, cough, and sore throat
                return fever && cough && soreThroat && temp < 102.0;
            },
            [](FactBase& facts) {
                facts.addFact(Fact("diagnosis", std::string("Common Cold")));
                facts.addFact(Fact("treatment", std::string("Rest, fluids, and over-the-counter cold medication")));
                std::cout << "Rule 'common_cold_rule' executed: Diagnosed with Common Cold." << std::endl;
            },
            10  // Lower priority
        ));
        
        // Rule 2: Flu
        diagnosisSystem.addRule(Rule(
            "flu_rule",
            [](const FactBase& facts) {
                auto hasFever = facts.getFact("has_fever");
                auto hasCough = facts.getFact("has_cough");
                auto hasJointPain = facts.getFact("has_joint_pain");
                
                if (!hasFever || !hasCough) {
                    return false;
                }
                
                bool fever = std::get<bool>(hasFever->getValue());
                bool cough = std::get<bool>(hasCough->getValue());
                bool jointPain = hasJointPain ? std::get<bool>(hasJointPain->getValue()) : false;
                
                auto bodyTemp = facts.getFact("body_temperature");
                double temp = bodyTemp ? std::get<double>(bodyTemp->getValue()) : 98.6;
                
                // Flu: high fever, cough, and sometimes joint pain
                return fever && cough && temp >= 102.0;
            },
            [](FactBase& facts) {
                facts.addFact(Fact("diagnosis", std::string("Influenza (Flu)")));
                facts.addFact(Fact("treatment", std::string("Rest, fluids, antiviral medications if caught early")));
                std::cout << "Rule 'flu_rule' executed: Diagnosed with Influenza." << std::endl;
            },
            20  // Higher priority than common cold
        ));
        
        // Rule 3: Allergic Reaction
        diagnosisSystem.addRule(Rule(
            "allergy_rule",
            [](const FactBase& facts) {
                auto hasRash = facts.getFact("has_rash");
                
                if (!hasRash) {
                    return false;
                }
                
                return std::get<bool>(hasRash->getValue());
            },
            [](FactBase& facts) {
                facts.addFact(Fact("diagnosis", std::string("Allergic Reaction")));
                facts.addFact(Fact("treatment", std::string("Antihistamines, identify and avoid allergen")));
                std::cout << "Rule 'allergy_rule' executed: Diagnosed with Allergic Reaction." << std::endl;
            },
            30  // Highest priority
        ));
        
        // Print initial symptoms
        std::cout << "Patient symptoms:" << std::endl;
        for (const auto& fact : diagnosisSystem.getFactBase()->getAllFacts()) {
            std::cout << "  " << fact.toString() << std::endl;
        }
        
        // Run the diagnosis
        std::cout << "\nRunning diagnosis..." << std::endl;
        size_t rulesExecuted = diagnosisSystem.run();
        std::cout << "Number of rules executed: " << rulesExecuted << std::endl;
        
        // Print diagnosis
        std::cout << "\nDiagnosis results:" << std::endl;
        auto diagnosis = diagnosisSystem.getFactBase()->getFact("diagnosis");
        auto treatment = diagnosisSystem.getFactBase()->getFact("treatment");
        
        if (diagnosis) {
            std::cout << "  Diagnosis: " << std::get<std::string>(diagnosis->getValue()) << std::endl;
        } else {
            std::cout << "  No diagnosis could be determined." << std::endl;
        }
        
        if (treatment) {
            std::cout << "  Recommended treatment: " << std::get<std::string>(treatment->getValue()) << std::endl;
        }
        
        // Update symptoms and run again
        std::cout << "\nUpdating symptoms: temperature rising to 103.0, developing joint pain..." << std::endl;
        diagnosisSystem.addFact(Fact("body_temperature", 103.0));
        diagnosisSystem.addFact(Fact("has_joint_pain", true));
        
        // Clear previous diagnosis
        diagnosisSystem.getFactBase()->removeFact("diagnosis");
        diagnosisSystem.getFactBase()->removeFact("treatment");
        
        // Run diagnosis again
        std::cout << "Running diagnosis again..." << std::endl;
        rulesExecuted = diagnosisSystem.run();
        std::cout << "Number of rules executed: " << rulesExecuted << std::endl;
        
        // Print updated diagnosis
        std::cout << "\nUpdated diagnosis results:" << std::endl;
        diagnosis = diagnosisSystem.getFactBase()->getFact("diagnosis");
        treatment = diagnosisSystem.getFactBase()->getFact("treatment");
        
        if (diagnosis) {
            std::cout << "  Diagnosis: " << std::get<std::string>(diagnosis->getValue()) << std::endl;
        } else {
            std::cout << "  No diagnosis could be determined." << std::endl;
        }
        
        if (treatment) {
            std::cout << "  Recommended treatment: " << std::get<std::string>(treatment->getValue()) << std::endl;
        }
    }
}

/**
 * @brief Main function to demonstrate the rule-based system
 */
int main() {
    try {
        // Run tests
        test::runSimpleTest();
        test::runMedicalDiagnosisTest();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}