/**
 * @file test_StaticCodeAnalyzer.cpp
 * @brief Unit tests for StaticCodeAnalyzer
 */

#include "StaticCodeAnalyzer.hpp"
#include <gtest/gtest.h>

using namespace StaticCodeAnalyzer;

TEST(StaticCodeAnalyzer, LongFunctionDetected) {
    std::vector<std::string> lines;
    lines.push_back("void Foo() {");
    for (int i = 0; i < 60; ++i) lines.push_back("  int x = 1;");
    lines.push_back("}");

    LongFunctionRule rule(50);
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_EQ(issues.size(), 1u);
    EXPECT_EQ(issues[0].SeverityLevel, Severity::WARNING);
}

TEST(StaticCodeAnalyzer, ShortFunctionPasses) {
    std::vector<std::string> lines = {"void Foo() {", "  return;", "}"};
    LongFunctionRule rule(50);
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_TRUE(issues.empty());
}

TEST(StaticCodeAnalyzer, MemoryLeakDetected) {
    std::vector<std::string> lines = {
        "int* p = new int;",
        "int* q = new int;",
        "delete p;"
    };
    MemoryLeakRule rule;
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_EQ(issues.size(), 1u);
    EXPECT_EQ(issues[0].SeverityLevel, Severity::ERROR);
}

TEST(StaticCodeAnalyzer, NoMemoryLeak) {
    std::vector<std::string> lines = {"int* p = new int;", "delete p;"};
    MemoryLeakRule rule;
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_TRUE(issues.empty());
}

TEST(StaticCodeAnalyzer, NamingConventionMixed) {
    std::vector<std::string> lines = {"int Some_Variable = 0;"};
    NamingConventionRule rule;
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_GE(issues.size(), 1u);
}

TEST(StaticCodeAnalyzer, NamingConventionClean) {
    std::vector<std::string> lines = {"int someVariable = 0;"};
    NamingConventionRule rule;
    auto issues = rule.Check("test.cpp", lines);
    EXPECT_TRUE(issues.empty());
}

TEST(StaticCodeAnalyzer, EmptyFileNoIssues) {
    std::vector<std::string> lines;
    LongFunctionRule rule(50);
    auto issues = rule.Check("empty.cpp", lines);
    EXPECT_TRUE(issues.empty());
}
