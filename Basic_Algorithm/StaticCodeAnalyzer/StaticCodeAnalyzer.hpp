/**
 * @file StaticCodeAnalyzer.hpp
 * @brief Lightweight static code analyzer for C++ lint rules
 * @details Parses C++ source text to detect common issues: long functions,
 *          potential memory leaks, null pointer dereferences, mutex misuse,
 *          and naming convention violations.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <map>
#include <set>
#include <filesystem>
#include <thread>
#include <mutex>
#include <future>
#include <memory>

namespace Fs = std::filesystem;

namespace StaticCodeAnalyzer {

enum class Severity { INFO, WARNING, ERROR, CRITICAL };

struct CodeIssue {
    std::string Filename;
    int LineNumber;
    Severity SeverityLevel;
    std::string RuleId;
    std::string Message;
    std::string CodeSnippet;
};

// ===== Base Rule =====
class AnalysisRule {
protected:
    std::string ruleId_;
    std::string description_;
    Severity severityLevel_;

public:
    AnalysisRule(const std::string& id, const std::string& desc, Severity sev)
        : ruleId_(id), description_(desc), severityLevel_(sev) {}
    virtual ~AnalysisRule() = default;

    virtual std::vector<CodeIssue> Check(const std::string& filename,
                                         const std::vector<std::string>& lines) = 0;

    std::string GetId() const { return ruleId_; }
    std::string GetDescription() const { return description_; }
    Severity GetSeverity() const { return severityLevel_; }
};

// ===== Long Function Rule =====
class LongFunctionRule : public AnalysisRule {
    int maxLines_;
public:
    explicit LongFunctionRule(int maxLines = 50)
        : AnalysisRule("FUNC_LENGTH", "Function exceeds maximum recommended length",
                       Severity::WARNING),
          maxLines_(maxLines) {}

    std::vector<CodeIssue> Check(const std::string& filename,
                                 const std::vector<std::string>& lines) override;
};

// ===== Memory Leak Rule =====
class MemoryLeakRule : public AnalysisRule {
public:
    MemoryLeakRule()
        : AnalysisRule("MEM_LEAK", "Potential memory leak with raw pointer",
                       Severity::ERROR) {}

    std::vector<CodeIssue> Check(const std::string& filename,
                                 const std::vector<std::string>& lines) override;
};

// ===== Null Pointer Dereference Rule =====
class NullPointerDereferenceRule : public AnalysisRule {
public:
    NullPointerDereferenceRule()
        : AnalysisRule("NULL_DEREF", "Potential null pointer dereference",
                       Severity::ERROR) {}

    std::vector<CodeIssue> Check(const std::string& filename,
                                 const std::vector<std::string>& lines) override;
};

// ===== Mutex Usage Rule =====
class MutexUsageRule : public AnalysisRule {
public:
    MutexUsageRule()
        : AnalysisRule("MUTEX_USAGE", "Improper mutex usage",
                       Severity::WARNING) {}

    std::vector<CodeIssue> Check(const std::string& filename,
                                 const std::vector<std::string>& lines) override;
};

// ===== Naming Convention Rule =====
class NamingConventionRule : public AnalysisRule {
public:
    NamingConventionRule()
        : AnalysisRule("NAMING", "Variable naming convention violation",
                       Severity::INFO) {}

    std::vector<CodeIssue> Check(const std::string& filename,
                                 const std::vector<std::string>& lines) override;
};

// ===== Analyzer =====
class Analyzer {
public:
    void RegisterRules();
    std::vector<std::string> ReadFileLines(const std::string& filename);
    std::vector<CodeIssue> AnalyzeFile(const std::string& filename);
    std::vector<CodeIssue> AnalyzeFiles(const std::vector<std::string>& filenames);
    std::vector<std::string> FindCppFiles(const std::string& directory);
    void PrintIssues(const std::vector<CodeIssue>& issues);
    void ExportToJson(const std::vector<CodeIssue>& issues, const std::string& filename);

private:
    std::vector<std::unique_ptr<AnalysisRule>> rules_;
    std::mutex issuesMutex_;
};

void PrintUsage();

} // namespace StaticCodeAnalyzer
