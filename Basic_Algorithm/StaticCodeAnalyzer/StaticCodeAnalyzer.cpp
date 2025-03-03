/**
 * @file StaticCodeAnalyzer.cpp
 * @brief Implementation of static code analyzer rules
 */

#include "StaticCodeAnalyzer.hpp"

namespace StaticCodeAnalyzer {

// ===== LongFunctionRule =====
std::vector<CodeIssue> LongFunctionRule::Check(const std::string& filename,
                                                const std::vector<std::string>& lines) {
    std::vector<CodeIssue> issues;
    int braceCount = 0;
    int funcStartLine = -1;

    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        for (char c : lines[i]) {
            if (c == '{') {
                if (braceCount == 0) funcStartLine = i;
                braceCount++;
            } else if (c == '}') {
                braceCount--;
                if (braceCount == 0 && funcStartLine >= 0) {
                    int funcLength = i - funcStartLine + 1;
                    if (funcLength > maxLines_) {
                        issues.push_back({filename, funcStartLine + 1, severityLevel_,
                                          ruleId_,
                                          "Function length of " + std::to_string(funcLength) +
                                          " lines exceeds maximum of " + std::to_string(maxLines_),
                                          lines[funcStartLine]});
                    }
                    funcStartLine = -1;
                }
            }
        }
    }
    return issues;
}

// ===== MemoryLeakRule =====
std::vector<CodeIssue> MemoryLeakRule::Check(const std::string& filename,
                                              const std::vector<std::string>& lines) {
    std::vector<CodeIssue> issues;
    std::regex newRegex(R"((\w+)\s*=\s*new\s+\w+)");
    std::regex deleteRegex(R"(delete\s+(\w+))");

    std::map<std::string, int> allocations;

    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        std::smatch match;
        std::string line = lines[i];

        while (std::regex_search(line, match, newRegex)) {
            allocations[match[1]] = i;
            line = match.suffix();
        }

        line = lines[i];
        while (std::regex_search(line, match, deleteRegex)) {
            allocations.erase(match[1]);
            line = match.suffix();
        }
    }

    for (const auto& [var, lineNum] : allocations) {
        issues.push_back({filename, lineNum + 1, severityLevel_, ruleId_,
                          "Potential memory leak: " + var,
                          lines[lineNum]});
    }
    return issues;
}

// ===== NullPointerDereferenceRule =====
std::vector<CodeIssue> NullPointerDereferenceRule::Check(const std::string& filename,
                                                          const std::vector<std::string>& lines) {
    std::vector<CodeIssue> issues;
    std::map<std::string, bool> pointers;
    std::regex ptrDeclRegex(R"((\w+)\s*\*\s*(\w+))");
    std::regex nullCheckRegex(R"(if\s*\(\s*(\w+)\s*(!?=|==)\s*(nullptr|NULL|0)\s*\))");
    std::regex derefRegex(R"((\w+)\s*(?:->|\[|\*))");

    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        std::smatch match;
        std::string line = lines[i];

        while (std::regex_search(line, match, ptrDeclRegex)) {
            pointers[match[2]] = false;
            line = match.suffix();
        }

        line = lines[i];
        while (std::regex_search(line, match, nullCheckRegex)) {
            pointers[match[1]] = true;
            line = match.suffix();
        }

        line = lines[i];
        while (std::regex_search(line, match, derefRegex)) {
            std::string ptrName = match[1];
            if (pointers.count(ptrName) && !pointers[ptrName]) {
                issues.push_back({filename, i + 1, severityLevel_, ruleId_,
                                  "Potential null pointer dereference: " + ptrName,
                                  lines[i]});
            }
            line = match.suffix();
        }
    }
    return issues;
}

// ===== MutexUsageRule =====
std::vector<CodeIssue> MutexUsageRule::Check(const std::string& filename,
                                              const std::vector<std::string>& lines) {
    std::vector<CodeIssue> issues;
    std::regex lockRegex(R"((\w+)\.lock\(\))");
    std::regex unlockRegex(R"((\w+)\.unlock\(\))");
    std::regex lockGuardRegex(R"(std::lock_guard<[^>]*>\s+\w+\s*\(\s*(\w+)\s*\))");

    std::map<std::string, int> locks;
    std::set<std::string> safeMutexes;

    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        std::smatch match;
        std::string line = lines[i];

        while (std::regex_search(line, match, lockGuardRegex)) {
            safeMutexes.insert(match[1]);
            line = match.suffix();
        }

        line = lines[i];
        while (std::regex_search(line, match, lockRegex)) {
            if (safeMutexes.find(match[1]) == safeMutexes.end())
                locks[match[1]] = i;
            line = match.suffix();
        }

        line = lines[i];
        while (std::regex_search(line, match, unlockRegex)) {
            locks.erase(match[1]);
            line = match.suffix();
        }
    }

    for (const auto& [mtx, lineNum] : locks) {
        issues.push_back({filename, lineNum + 1, severityLevel_, ruleId_,
                          "Potential mutex issue: " + mtx + " locked but not unlocked",
                          lines[lineNum]});
    }
    return issues;
}

// ===== NamingConventionRule =====
std::vector<CodeIssue> NamingConventionRule::Check(const std::string& filename,
                                                    const std::vector<std::string>& lines) {
    std::vector<CodeIssue> issues;
    std::regex varDeclRegex(R"((\w+)\s+(\w+)\s*[;=])");

    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        std::smatch match;
        std::string line = lines[i];

        while (std::regex_search(line, match, varDeclRegex)) {
            std::string varName = match[2];
            bool hasUppercase = false, hasUnderscore = false;
            for (char c : varName) {
                if (std::isupper(c)) hasUppercase = true;
                if (c == '_') hasUnderscore = true;
            }
            if (hasUppercase && hasUnderscore) {
                issues.push_back({filename, i + 1, severityLevel_, ruleId_,
                                  "Variable '" + varName + "' uses mixed naming convention",
                                  lines[i]});
            }
            line = match.suffix();
        }
    }
    return issues;
}

// ===== Analyzer =====
void Analyzer::RegisterRules() {
    rules_.push_back(std::make_unique<LongFunctionRule>());
    rules_.push_back(std::make_unique<NullPointerDereferenceRule>());
    rules_.push_back(std::make_unique<MemoryLeakRule>());
    rules_.push_back(std::make_unique<MutexUsageRule>());
    rules_.push_back(std::make_unique<NamingConventionRule>());
}

std::vector<std::string> Analyzer::ReadFileLines(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
    std::string line;
    while (std::getline(file, line)) lines.push_back(line);
    return lines;
}

std::vector<CodeIssue> Analyzer::AnalyzeFile(const std::string& filename) {
    std::vector<CodeIssue> fileIssues;
    try {
        auto lines = ReadFileLines(filename);
        for (const auto& rule : rules_) {
            auto ruleIssues = rule->Check(filename, lines);
            fileIssues.insert(fileIssues.end(), ruleIssues.begin(), ruleIssues.end());
        }
    } catch (const std::exception& e) {
        std::cerr << "Error analyzing " << filename << ": " << e.what() << std::endl;
    }
    return fileIssues;
}

std::vector<CodeIssue> Analyzer::AnalyzeFiles(const std::vector<std::string>& filenames) {
    std::vector<CodeIssue> allIssues;
    std::vector<std::future<std::vector<CodeIssue>>> futures;

    for (const auto& f : filenames) {
        futures.push_back(std::async(std::launch::async,
                                     [this, f]() { return AnalyzeFile(f); }));
    }

    for (auto& future : futures) {
        auto fileIssues = future.get();
        std::lock_guard<std::mutex> lock(issuesMutex_);
        allIssues.insert(allIssues.end(), fileIssues.begin(), fileIssues.end());
    }
    return allIssues;
}

std::vector<std::string> Analyzer::FindCppFiles(const std::string& directory) {
    std::vector<std::string> filenames;
    try {
        for (const auto& entry : Fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".cpp" || ext == ".h" || ext == ".hpp")
                    filenames.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }
    return filenames;
}

void Analyzer::PrintIssues(const std::vector<CodeIssue>& issues) {
    std::map<std::string, std::vector<CodeIssue>> byFile;
    for (const auto& issue : issues) byFile[issue.Filename].push_back(issue);

    std::cout << "Total issues: " << issues.size() << std::endl;
    for (const auto& [filename, fileIssues] : byFile) {
        std::cout << "File: " << filename << std::endl;
        for (const auto& issue : fileIssues) {
            std::cout << "  [" << issue.RuleId << "] Line " << issue.LineNumber
                      << ": " << issue.Message << std::endl;
        }
    }
}

void Analyzer::ExportToJson(const std::vector<CodeIssue>& issues, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    file << "{\n  \"issues\": [\n";
    for (std::size_t i = 0; i < issues.size(); ++i) {
        const auto& issue = issues[i];
        file << "    {\"filename\": \"" << issue.Filename
             << "\", \"line\": " << issue.LineNumber
             << ", \"rule\": \"" << issue.RuleId
             << "\", \"message\": \"" << issue.Message << "\"}";
        if (i < issues.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n}\n";
}

void PrintUsage() {
    std::cout << "Usage: static_analyzer [options] <path>" << std::endl;
    std::cout << "  --help     Show help" << std::endl;
    std::cout << "  --json F   Export to JSON" << std::endl;
    std::cout << "  --verbose  Detailed output" << std::endl;
}

} // namespace StaticCodeAnalyzer
