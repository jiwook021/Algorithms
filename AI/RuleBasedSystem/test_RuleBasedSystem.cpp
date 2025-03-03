/**
 * @file test_RuleBasedSystem.cpp
 * @brief Google Test suite for the RuleBasedSystem module.
 *
 * Covers: Fact, FactBase, Rule, RuleBase, InferenceEngine, and
 *         the RuleBasedSystem facade.  Scenarios include forward
 *         chaining, rule priority, fixpoint termination, and a
 *         medical-diagnosis integration test.
 */

#include <gtest/gtest.h>
#include "RuleBasedSystem.hpp"

#include <iostream>
#include <string>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

}  // namespace

// ======================== Fact ========================

TEST(Fact, CreationAndAccess) {
    PrintSection("Fact Creation And Access",
        "Verifies that a Fact stores its name and typed value correctly");

    Fact f("temperature", 25.0);
    EXPECT_EQ(f.GetName(), "temperature");
    EXPECT_DOUBLE_EQ(std::get<double>(f.GetValue()), 25.0);
}

TEST(Fact, SetValue) {
    PrintSection("Fact SetValue",
        "Verifies that SetValue replaces the stored variant value");

    Fact f("x", 1);
    f.SetValue(42);
    EXPECT_EQ(std::get<int>(f.GetValue()), 42);
}

TEST(Fact, ToStringBool) {
    PrintSection("Fact ToString Bool",
        "Verifies the Overloaded visitor renders booleans as true/false");

    Fact f("active", true);
    EXPECT_NE(f.ToString().find("true"), std::string::npos);
}

TEST(Fact, ToStringString) {
    PrintSection("Fact ToString String",
        "Verifies the Overloaded visitor renders strings with quotes");

    Fact f("name", std::string("test"));
    EXPECT_NE(f.ToString().find("test"), std::string::npos);
}

TEST(Fact, ToStringInt) {
    PrintSection("Fact ToString Int",
        "Verifies the Overloaded visitor renders integers correctly");

    Fact f("count", 7);
    std::string s = f.ToString();
    EXPECT_NE(s.find("7"), std::string::npos);
}

TEST(Fact, ToStringDouble) {
    PrintSection("Fact ToString Double",
        "Verifies the Overloaded visitor renders doubles correctly");

    Fact f("pi", 3.14);
    std::string s = f.ToString();
    EXPECT_NE(s.find("3.14"), std::string::npos);
}

// ======================== FactBase ========================

TEST(FactBase, AddAndHas) {
    PrintSection("FactBase Add And Has",
        "Verifies facts can be added and queried by name");

    FactBase fb;
    EXPECT_TRUE(fb.AddFact(Fact("temp", 25.0)));
    EXPECT_TRUE(fb.HasFact("temp"));
    EXPECT_FALSE(fb.HasFact("missing"));
}

TEST(FactBase, AddDuplicateUpdates) {
    PrintSection("FactBase Add Duplicate Updates",
        "Verifies adding a fact with the same name updates (returns false) without growing size");

    FactBase fb;
    EXPECT_TRUE(fb.AddFact(Fact("x", 1)));
    EXPECT_FALSE(fb.AddFact(Fact("x", 2)));
    EXPECT_EQ(fb.size(), 1u);
}

TEST(FactBase, GetFact) {
    PrintSection("FactBase GetFact",
        "Verifies GetFact returns the fact or nullopt");

    FactBase fb;
    fb.AddFact(Fact("speed", 100.0));
    auto f = fb.GetFact("speed");
    EXPECT_TRUE(f.has_value());
    EXPECT_EQ(f->GetName(), "speed");
    EXPECT_FALSE(fb.GetFact("missing").has_value());
}

TEST(FactBase, RemoveFact) {
    PrintSection("FactBase RemoveFact",
        "Verifies facts can be removed and double-removal returns false");

    FactBase fb;
    fb.AddFact(Fact("x", 1));
    EXPECT_TRUE(fb.HasFact("x"));
    EXPECT_TRUE(fb.RemoveFact("x"));
    EXPECT_FALSE(fb.HasFact("x"));
    EXPECT_FALSE(fb.RemoveFact("x"));
}

TEST(FactBase, ClearAndSize) {
    PrintSection("FactBase Clear And Size",
        "Verifies clear() empties the fact base");

    FactBase fb;
    fb.AddFact(Fact("a", 1));
    fb.AddFact(Fact("b", 2));
    EXPECT_EQ(fb.size(), 2u);
    fb.clear();
    EXPECT_EQ(fb.size(), 0u);
}

TEST(FactBase, GetAllFacts) {
    PrintSection("FactBase GetAllFacts",
        "Verifies GetAllFacts returns every stored fact");

    FactBase fb;
    fb.AddFact(Fact("a", 1));
    fb.AddFact(Fact("b", 2));
    auto all = fb.GetAllFacts();
    EXPECT_EQ(all.size(), 2u);
}

// ======================== Rule ========================

TEST(Rule, EvaluateCondition) {
    PrintSection("Rule Evaluate Condition",
        "Verifies a rule's condition lambda fires when facts match");

    FactBase fb;
    fb.AddFact(Fact("temp", 90.0));

    Rule rule("hot",
        [](const FactBase& facts) {
            auto t = facts.GetFact("temp");
            return t && std::get<double>(t->GetValue()) > 80.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("comfort", std::string("hot")));
        });

    EXPECT_TRUE(rule.Evaluate(fb));
}

TEST(Rule, ExecuteAction) {
    PrintSection("Rule Execute Action",
        "Verifies a rule's action lambda modifies the fact base");

    FactBase fb;

    Rule rule("add_result",
        [](const FactBase&) { return true; },
        [](FactBase& facts) {
            facts.AddFact(Fact("result", std::string("done")));
        });

    rule.Execute(fb);
    EXPECT_TRUE(fb.HasFact("result"));
}

TEST(Rule, NullConditionReturnsFalse) {
    PrintSection("Rule Null Condition Returns False",
        "Verifies that a rule with a null condition evaluates to false");

    Rule rule("null_cond", nullptr, nullptr, 0);
    FactBase fb;
    EXPECT_FALSE(rule.Evaluate(fb));
}

TEST(Rule, Priority) {
    PrintSection("Rule Priority",
        "Verifies priority is stored and comparable");

    Rule low("low", nullptr, nullptr, 1);
    Rule high("high", nullptr, nullptr, 10);
    EXPECT_GT(high.GetPriority(), low.GetPriority());
}

// ======================== RuleBase ========================

TEST(RuleBase, AddAndRemove) {
    PrintSection("RuleBase Add And Remove",
        "Verifies rules can be added (no duplicates) and removed");

    RuleBase rb;
    Rule r("test", [](const FactBase&) { return true; }, [](FactBase&) {});
    EXPECT_TRUE(rb.AddRule(r));
    EXPECT_FALSE(rb.AddRule(r));  // duplicate name
    EXPECT_TRUE(rb.RemoveRule("test"));
    EXPECT_EQ(rb.size(), 0u);
}

TEST(RuleBase, GetAllRulesSortedByPriority) {
    PrintSection("RuleBase GetAllRules Sorted By Priority",
        "Verifies rules come back sorted highest-priority first");

    RuleBase rb;
    rb.AddRule(Rule("low", [](const FactBase&) { return true; }, [](FactBase&) {}, 1));
    rb.AddRule(Rule("high", [](const FactBase&) { return true; }, [](FactBase&) {}, 10));
    auto rules = rb.GetAllRules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_GT(rules[0].GetPriority(), rules[1].GetPriority());
}

// ======================== InferenceEngine ========================

TEST(InferenceEngine, NullRuleBaseThrows) {
    PrintSection("InferenceEngine Null RuleBase Throws",
        "Verifies the constructor rejects a null RuleBase pointer");

    auto fb = std::make_shared<FactBase>();
    EXPECT_THROW(InferenceEngine(nullptr, fb), std::invalid_argument);
}

TEST(InferenceEngine, NullFactBaseThrows) {
    PrintSection("InferenceEngine Null FactBase Throws",
        "Verifies the constructor rejects a null FactBase pointer");

    auto rb = std::make_shared<RuleBase>();
    EXPECT_THROW(InferenceEngine(rb, nullptr), std::invalid_argument);
}

TEST(InferenceEngine, ForwardChainingDerivesNewFacts) {
    PrintSection("Forward Chaining Derives New Facts",
        "Verifies the inference engine fires rules to derive facts not in the initial fact base");

    auto rb = std::make_shared<RuleBase>();
    auto fb = std::make_shared<FactBase>();

    fb->AddFact(Fact("a", true));

    // Rule: if 'a' exists, derive 'b'
    rb->AddRule(Rule("derive_b",
        [](const FactBase& facts) { return facts.HasFact("a"); },
        [](FactBase& facts) { facts.AddFact(Fact("b", true)); }));

    // Rule: if 'b' exists, derive 'c'  (chain)
    rb->AddRule(Rule("derive_c",
        [](const FactBase& facts) { return facts.HasFact("b"); },
        [](FactBase& facts) { facts.AddFact(Fact("c", true)); }));

    InferenceEngine engine(rb, fb);
    size_t fired = engine.Run();

    EXPECT_GE(fired, 2u);
    EXPECT_TRUE(fb->HasFact("b"));
    EXPECT_TRUE(fb->HasFact("c"));
}

TEST(InferenceEngine, RulePriorityDeterminesExecutionOrder) {
    PrintSection("Rule Priority Determines Execution Order",
        "Verifies higher-priority rules fire before lower-priority ones within the same cycle");

    auto rb = std::make_shared<RuleBase>();
    auto fb = std::make_shared<FactBase>();

    fb->AddFact(Fact("start", true));

    // Low priority: sets label to "low" only if no label yet
    rb->AddRule(Rule("low_priority",
        [](const FactBase& facts) {
            return facts.HasFact("start") && !facts.HasFact("label");
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("label", std::string("low")));
        },
        1));

    // High priority: sets label to "high" only if no label yet
    rb->AddRule(Rule("high_priority",
        [](const FactBase& facts) {
            return facts.HasFact("start") && !facts.HasFact("label");
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("label", std::string("high")));
        },
        10));

    InferenceEngine engine(rb, fb);
    engine.Run();

    auto label = fb->GetFact("label");
    ASSERT_TRUE(label.has_value());
    EXPECT_EQ(std::get<std::string>(label->GetValue()), "high");
}

TEST(InferenceEngine, FixpointTermination) {
    PrintSection("Fixpoint Termination",
        "Verifies the engine stops when no new facts are derived (fixpoint)");

    auto rb = std::make_shared<RuleBase>();
    auto fb = std::make_shared<FactBase>();

    fb->AddFact(Fact("x", true));

    // This rule derives 'y' from 'x'.  Once 'y' exists, nothing new can happen.
    rb->AddRule(Rule("x_to_y",
        [](const FactBase& facts) { return facts.HasFact("x"); },
        [](FactBase& facts) { facts.AddFact(Fact("y", true)); }));

    InferenceEngine engine(rb, fb, /*max_iterations=*/1000);
    size_t fired = engine.Run();

    // The rule fires exactly once; the engine should stop after that cycle.
    EXPECT_EQ(fired, 1u);
    EXPECT_TRUE(fb->HasFact("y"));
}

TEST(InferenceEngine, MaxIterationsCapsExecution) {
    PrintSection("Max Iterations Caps Execution",
        "Verifies the engine respects the iteration cap to prevent infinite loops");

    auto rb = std::make_shared<RuleBase>();
    auto fb = std::make_shared<FactBase>();

    fb->AddFact(Fact("counter", 0));

    InferenceEngine engine(rb, fb, 5);
    EXPECT_EQ(engine.GetMaxIterations(), 5u);

    engine.SetMaxIterations(10);
    EXPECT_EQ(engine.GetMaxIterations(), 10u);
}

// ======================== RuleBasedSystem (facade) ========================

TEST(RuleBasedSystem, EndToEnd) {
    PrintSection("RuleBasedSystem End-To-End",
        "Verifies the facade wires fact base, rule base, and engine correctly");

    RuleBasedSystem sys("test");
    sys.AddFact(Fact("temp", 90.0));
    sys.AddRule(Rule("hot",
        [](const FactBase& facts) {
            auto t = facts.GetFact("temp");
            return t && std::get<double>(t->GetValue()) > 80.0;
        },
        [](FactBase& facts) {
            facts.AddFact(Fact("hot", true));
        }));

    size_t fired = sys.Run();
    EXPECT_GE(fired, 1u);
    EXPECT_TRUE(sys.GetFactBase()->HasFact("hot"));
}

TEST(RuleBasedSystem, AccessorsReturnValidPointers) {
    PrintSection("RuleBasedSystem Accessors",
        "Verifies GetFactBase, GetRuleBase, GetInferenceEngine return non-null");

    RuleBasedSystem sys("acc");
    EXPECT_NE(sys.GetFactBase(), nullptr);
    EXPECT_NE(sys.GetRuleBase(), nullptr);
    EXPECT_NE(sys.GetInferenceEngine(), nullptr);
    EXPECT_EQ(sys.GetName(), "acc");
}

// ======================== Medical Diagnosis Scenario ========================

TEST(RuleBasedSystem, MedicalDiagnosisCommonCold) {
    PrintSection("Medical Diagnosis -- Common Cold",
        "Verifies the system diagnoses a common cold with mild fever, cough, and sore throat");

    RuleBasedSystem diag("MedDiag");

    diag.AddFact(Fact("has_fever", true));
    diag.AddFact(Fact("body_temperature", 101.0));
    diag.AddFact(Fact("has_cough", true));
    diag.AddFact(Fact("has_sore_throat", true));
    diag.AddFact(Fact("has_rash", false));

    // Common Cold rule (priority 10)
    diag.AddRule(Rule("common_cold",
        [](const FactBase& facts) {
            auto fever  = facts.GetFact("has_fever");
            auto cough  = facts.GetFact("has_cough");
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
        },
        10));

    // Flu rule (priority 20) -- should NOT fire because temp < 102
    diag.AddRule(Rule("flu",
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
            facts.AddFact(Fact("diagnosis", std::string("Influenza")));
        },
        20));

    diag.Run();

    auto dx = diag.GetFactBase()->GetFact("diagnosis");
    ASSERT_TRUE(dx.has_value());
    EXPECT_EQ(std::get<std::string>(dx->GetValue()), "Common Cold");
}

TEST(RuleBasedSystem, MedicalDiagnosisFlu) {
    PrintSection("Medical Diagnosis -- Flu",
        "Verifies the system diagnoses flu when temperature >= 102");

    RuleBasedSystem diag("MedDiag");

    diag.AddFact(Fact("has_fever", true));
    diag.AddFact(Fact("body_temperature", 103.0));
    diag.AddFact(Fact("has_cough", true));
    diag.AddFact(Fact("has_sore_throat", true));
    diag.AddFact(Fact("has_rash", false));

    // Common Cold (priority 10)
    diag.AddRule(Rule("common_cold",
        [](const FactBase& facts) {
            auto fever  = facts.GetFact("has_fever");
            auto cough  = facts.GetFact("has_cough");
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
        },
        10));

    // Flu (priority 20)
    diag.AddRule(Rule("flu",
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
            facts.AddFact(Fact("diagnosis", std::string("Influenza")));
        },
        20));

    diag.Run();

    auto dx = diag.GetFactBase()->GetFact("diagnosis");
    ASSERT_TRUE(dx.has_value());
    // Flu rule has higher priority and fires first because temp >= 102
    EXPECT_EQ(std::get<std::string>(dx->GetValue()), "Influenza");
}
