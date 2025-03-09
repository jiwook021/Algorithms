/**
 * StaticCodeAnalyzer - A tool to analyze C++ source code for potential issues
 * 
 * This program parses C++ source files and performs various static analysis checks
 * to identify common programming errors and code quality issues.
 */

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
 #include <atomic>
 #include <memory>
 
 namespace fs = std::filesystem;
 
 // Severity levels for issues
 enum class Severity {
     INFO,
     WARNING,
     ERROR,
     CRITICAL
 };
 
 // Issue representation
 struct CodeIssue {
     std::string filename;
     int line_number;
     Severity severity;
     std::string rule_id;
     std::string message;
     std::string code_snippet;
 };
 
 // Base class for all analysis rules
 class AnalysisRule {
 protected:
     std::string rule_id;
     std::string description;
     Severity severity;
 
 public:
     AnalysisRule(const std::string& id, const std::string& desc, Severity sev) 
         : rule_id(id), description(desc), severity(sev) {}
     
     virtual ~AnalysisRule() = default;
     
     // Each derived rule will implement this method to perform its specific check
     virtual std::vector<CodeIssue> check(const std::string& filename, 
                                        const std::vector<std::string>& lines) = 0;
     
     std::string getId() const { return rule_id; }
     std::string getDescription() const { return description; }
     Severity getSeverity() const { return severity; }
 };
 
 // Rule: Check for unusually long functions (potential complexity issue)
 class LongFunctionRule : public AnalysisRule {
     int max_lines;
 public:
     LongFunctionRule(int max = 50) 
         : AnalysisRule("FUNC_LENGTH", "Function exceeds maximum recommended length", Severity::WARNING),
           max_lines(max) {}
     
     std::vector<CodeIssue> check(const std::string& filename, 
                                const std::vector<std::string>& lines) override {
         std::vector<CodeIssue> issues;
         
         std::regex func_start_regex(R"(^\s*\w+\s+\w+\s*\([^)]*\)\s*(\{)?\s*$)");
         
         int brace_count = 0;
         int func_start_line = -1;
         
         for (int i = 0; i < lines.size(); ++i) {
             if (std::regex_search(lines[i], func_start_regex)) {
                 func_start_line = i;
                 brace_count = 0;
             }
             
             // Count opening and closing braces
             for (char c : lines[i]) {
                 if (c == '{') brace_count++;
                 else if (c == '}') {
                     brace_count--;
                     // Function end detected
                     if (brace_count == 0 && func_start_line != -1) {
                         int func_length = i - func_start_line + 1;
                         if (func_length > max_lines) {
                             std::string snippet = func_start_line >= 0 ? lines[func_start_line] : "";
                             issues.push_back({
                                 filename,
                                 func_start_line + 1, // 1-based line numbers
                                 severity,
                                 rule_id,
                                 "Function length of " + std::to_string(func_length) + 
                                 " lines exceeds maximum recommended length of " + 
                                 std::to_string(max_lines) + " lines",
                                 snippet
                             });
                         }
                         func_start_line = -1;
                     }
                 }
             }
         }
         
         return issues;
     }
 };
 
 // Rule: Check for potential null pointer dereferences
 class NullPointerDereferenceRule : public AnalysisRule {
 public:
     NullPointerDereferenceRule() 
         : AnalysisRule("NULL_DEREF", "Potential null pointer dereference", Severity::ERROR) {}
     
     std::vector<CodeIssue> check(const std::string& filename, 
                                const std::vector<std::string>& lines) override {
         std::vector<CodeIssue> issues;
         
         // Detect pointer variables
         std::map<std::string, bool> pointers;  // pointer name -> checked status
         std::regex ptr_decl_regex(R"((\w+)\s*\*\s*(\w+))");
         std::regex smart_ptr_regex(R"(std::(\w+)_ptr<[^>]*>\s+(\w+))");
         
         // Detect null checks
         std::regex null_check_regex(R"(if\s*\(\s*(\w+)\s*(!?=|==)\s*(nullptr|NULL|0)\s*\))");
         
         // Detect dereferences
         std::regex deref_regex(R"((\w+)\s*(?:->|\[|\*))");
         
         for (int i = 0; i < lines.size(); ++i) {
             // Find pointer declarations
             std::smatch match;
             std::string line = lines[i];
             
             // Raw pointer declarations
             while (std::regex_search(line, match, ptr_decl_regex)) {
                 pointers[match[2]] = false;  // Unchecked by default
                 line = match.suffix();
             }
             
             // Smart pointer declarations
             line = lines[i];
             while (std::regex_search(line, match, smart_ptr_regex)) {
                 pointers[match[2]] = false;
                 line = match.suffix();
             }
             
             // Find null checks
             line = lines[i];
             while (std::regex_search(line, match, null_check_regex)) {
                 pointers[match[1]] = true;  // Marked as checked
                 line = match.suffix();
             }
             
             // Find dereferences
             line = lines[i];
             while (std::regex_search(line, match, deref_regex)) {
                 std::string ptr_name = match[1];
                 if (pointers.count(ptr_name) && !pointers[ptr_name]) {
                     issues.push_back({
                         filename,
                         i + 1,
                         severity,
                         rule_id,
                         "Potential null pointer dereference: " + ptr_name + 
                         " is used without a null check",
                         lines[i]
                     });
                 }
                 line = match.suffix();
             }
         }
         
         return issues;
     }
 };
 
 // Rule: Check for potential memory leaks with raw pointers
 class MemoryLeakRule : public AnalysisRule {
 public:
     MemoryLeakRule() 
         : AnalysisRule("MEM_LEAK", "Potential memory leak with raw pointer", Severity::ERROR) {}
     
     std::vector<CodeIssue> check(const std::string& filename, 
                                const std::vector<std::string>& lines) override {
         std::vector<CodeIssue> issues;
         
         // Check for new without corresponding delete
         std::regex new_regex(R"((\w+)\s*=\s*new\s+\w+)");
         std::regex delete_regex(R"(delete\s+(\w+))");
         
         std::map<std::string, int> allocations;  // variable -> line number
         
         for (int i = 0; i < lines.size(); ++i) {
             // Find allocations
             std::smatch match;
             std::string line = lines[i];
             
             while (std::regex_search(line, match, new_regex)) {
                 allocations[match[1]] = i;
                 line = match.suffix();
             }
             
             // Find deallocations
             line = lines[i];
             while (std::regex_search(line, match, delete_regex)) {
                 allocations.erase(match[1]);
                 line = match.suffix();
             }
         }
         
         // Report any remaining allocations as potential leaks
         for (const auto& [var, line_num] : allocations) {
             issues.push_back({
                 filename,
                 line_num + 1,
                 severity,
                 rule_id,
                 "Potential memory leak: " + var + " is allocated with 'new' but might not be deleted",
                 lines[line_num]
             });
         }
         
         return issues;
     }
 };
 
 // Rule: Check for proper usage of mutexes
 class MutexUsageRule : public AnalysisRule {
 public:
     MutexUsageRule() 
         : AnalysisRule("MUTEX_USAGE", "Improper mutex usage", Severity::WARNING) {}
     
     std::vector<CodeIssue> check(const std::string& filename, 
                                const std::vector<std::string>& lines) override {
         std::vector<CodeIssue> issues;
         
         // Check for lock without unlock
         std::regex lock_regex(R"((\w+)\.lock\(\))");
         std::regex unlock_regex(R"((\w+)\.unlock\(\))");
         std::regex lock_guard_regex(R"(std::lock_guard<[^>]*>\s+\w+\s*\(\s*(\w+)\s*\))");
         std::regex unique_lock_regex(R"(std::unique_lock<[^>]*>\s+\w+\s*\(\s*(\w+)\s*\))");
         
         std::map<std::string, int> locks;  // mutex -> line number
         std::set<std::string> safe_mutexes;  // Mutexes used with lock_guard or unique_lock
         
         for (int i = 0; i < lines.size(); ++i) {
             std::smatch match;
             std::string line = lines[i];
             
             // Find lock_guard/unique_lock usages (these are safe)
             while (std::regex_search(line, match, lock_guard_regex) || 
                    std::regex_search(line, match, unique_lock_regex)) {
                 safe_mutexes.insert(match[1]);
                 line = match.suffix();
             }
             
             // Find direct lock calls
             line = lines[i];
             while (std::regex_search(line, match, lock_regex)) {
                 if (safe_mutexes.find(match[1]) == safe_mutexes.end()) {
                     locks[match[1]] = i;
                 }
                 line = match.suffix();
             }
             
             // Find unlock calls
             line = lines[i];
             while (std::regex_search(line, match, unlock_regex)) {
                 locks.erase(match[1]);
                 line = match.suffix();
             }
         }
         
         // Report any remaining locks as potential issues
         for (const auto& [mutex, line_num] : locks) {
             issues.push_back({
                 filename,
                 line_num + 1,
                 severity,
                 rule_id,
                 "Potential mutex issue: " + mutex + " is locked but might not be unlocked",
                 lines[line_num]
             });
         }
         
         return issues;
     }
 };
 
 // Rule: Check for variable naming conventions
 class NamingConventionRule : public AnalysisRule {
 public:
     NamingConventionRule() 
         : AnalysisRule("NAMING", "Variable naming convention violation", Severity::INFO) {}
     
     std::vector<CodeIssue> check(const std::string& filename, 
                                const std::vector<std::string>& lines) override {
         std::vector<CodeIssue> issues;
         
         // Check for variables with inconsistent naming
         std::regex var_decl_regex(R"((\w+)\s+(\w+)\s*[;=])");
         
         for (int i = 0; i < lines.size(); ++i) {
             std::smatch match;
             std::string line = lines[i];
             
             while (std::regex_search(line, match, var_decl_regex)) {
                 std::string var_name = match[2];
                 
                 // Simple check: variables should be snake_case or camelCase
                 bool has_uppercase = false;
                 bool has_underscore = false;
                 
                 for (char c : var_name) {
                     if (std::isupper(c)) has_uppercase = true;
                     if (c == '_') has_underscore = true;
                 }
                 
                 // Mixed convention (both uppercase and underscore)
                 if (has_uppercase && has_underscore) {
                     issues.push_back({
                         filename,
                         i + 1,
                         severity,
                         rule_id,
                         "Variable '" + var_name + "' uses mixed naming convention",
                         lines[i]
                     });
                 }
                 
                 line = match.suffix();
             }
         }
         
         return issues;
     }
 };
 
 class StaticAnalyzer {
 private:
     std::vector<std::unique_ptr<AnalysisRule>> rules;
     std::mutex issues_mutex; // For thread-safe access to issues vector
     
 public:
     // Add all the analysis rules to be used
     void registerRules() {
         rules.push_back(std::make_unique<LongFunctionRule>());
         rules.push_back(std::make_unique<NullPointerDereferenceRule>());
         rules.push_back(std::make_unique<MemoryLeakRule>());
         rules.push_back(std::make_unique<MutexUsageRule>());
         rules.push_back(std::make_unique<NamingConventionRule>());
         
         // More rules can be added here
     }
     
     // Read file contents into a vector of lines
     std::vector<std::string> readFileLines(const std::string& filename) {
         std::vector<std::string> lines;
         std::ifstream file(filename);
         
         if (!file.is_open()) {
             throw std::runtime_error("Failed to open file: " + filename);
         }
         
         std::string line;
         while (std::getline(file, line)) {
             lines.push_back(line);
         }
         
         return lines;
     }
     
     // Analyze a single file
     std::vector<CodeIssue> analyzeFile(const std::string& filename) {
         std::vector<CodeIssue> file_issues;
         
         try {
             auto lines = readFileLines(filename);
             
             // Apply each rule to the file
             for (const auto& rule : rules) {
                 auto rule_issues = rule->check(filename, lines);
                 file_issues.insert(file_issues.end(), rule_issues.begin(), rule_issues.end());
             }
         } catch (const std::exception& e) {
             std::cerr << "Error analyzing file " << filename << ": " << e.what() << std::endl;
         }
         
         return file_issues;
     }
     
     // Analyze multiple files in parallel
     std::vector<CodeIssue> analyzeFiles(const std::vector<std::string>& filenames) {
         std::vector<CodeIssue> all_issues;
         std::vector<std::future<std::vector<CodeIssue>>> futures;
         
         // Process each file in a separate thread
         for (const auto& filename : filenames) {
             futures.push_back(std::async(std::launch::async, 
                                        [this, filename]() { return analyzeFile(filename); }));
         }
         
         // Collect results
         for (auto& future : futures) {
             auto file_issues = future.get();
             std::lock_guard<std::mutex> lock(issues_mutex); // Thread-safe access to all_issues
             all_issues.insert(all_issues.end(), file_issues.begin(), file_issues.end());
         }
         
         return all_issues;
     }
     
     // Find all C++ files in a directory (recursively)
     std::vector<std::string> findCppFiles(const std::string& directory) {
         std::vector<std::string> filenames;
         
         try {
             for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                 if (entry.is_regular_file()) {
                     std::string ext = entry.path().extension().string();
                     if (ext == ".cpp" || ext == ".h" || ext == ".hpp") {
                         filenames.push_back(entry.path().string());
                     }
                 }
             }
         } catch (const std::exception& e) {
             std::cerr << "Error scanning directory: " << e.what() << std::endl;
         }
         
         return filenames;
     }
     
     // Format and print issues
     void printIssues(const std::vector<CodeIssue>& issues) {
         // Group issues by file
         std::map<std::string, std::vector<CodeIssue>> issues_by_file;
         
         for (const auto& issue : issues) {
             issues_by_file[issue.filename].push_back(issue);
         }
         
         // Print summary counts
         std::map<Severity, int> severity_counts;
         for (const auto& issue : issues) {
             severity_counts[issue.severity]++;
         }
         
         std::cout << "Analysis completed. Found issues:" << std::endl;
         std::cout << "  Critical: " << severity_counts[Severity::CRITICAL] << std::endl;
         std::cout << "  Error: " << severity_counts[Severity::ERROR] << std::endl;
         std::cout << "  Warning: " << severity_counts[Severity::WARNING] << std::endl;
         std::cout << "  Info: " << severity_counts[Severity::INFO] << std::endl;
         std::cout << "  Total: " << issues.size() << std::endl;
         std::cout << std::endl;
         
         // Print issues by file
         for (const auto& [filename, file_issues] : issues_by_file) {
             std::cout << "File: " << filename << std::endl;
             
             for (const auto& issue : file_issues) {
                 std::string severity_str;
                 switch (issue.severity) {
                     case Severity::INFO: severity_str = "INFO"; break;
                     case Severity::WARNING: severity_str = "WARNING"; break;
                     case Severity::ERROR: severity_str = "ERROR"; break;
                     case Severity::CRITICAL: severity_str = "CRITICAL"; break;
                 }
                 
                 std::cout << "  [" << severity_str << "][" << issue.rule_id << "] Line " 
                           << issue.line_number << ": " << issue.message << std::endl;
                 std::cout << "    Code: " << issue.code_snippet << std::endl;
             }
             
             std::cout << std::endl;
         }
     }
     
     // Export issues to a JSON file
     void exportToJson(const std::vector<CodeIssue>& issues, const std::string& filename) {
         std::ofstream file(filename);
         
         if (!file.is_open()) {
             std::cerr << "Failed to open output file: " << filename << std::endl;
             return;
         }
         
         file << "{\n  \"issues\": [\n";
         
         for (size_t i = 0; i < issues.size(); ++i) {
             const auto& issue = issues[i];
             
             std::string severity_str;
             switch (issue.severity) {
                 case Severity::INFO: severity_str = "INFO"; break;
                 case Severity::WARNING: severity_str = "WARNING"; break;
                 case Severity::ERROR: severity_str = "ERROR"; break;
                 case Severity::CRITICAL: severity_str = "CRITICAL"; break;
             }
             
             file << "    {\n";
             file << "      \"filename\": \"" << issue.filename << "\",\n";
             file << "      \"line\": " << issue.line_number << ",\n";
             file << "      \"severity\": \"" << severity_str << "\",\n";
             file << "      \"rule_id\": \"" << issue.rule_id << "\",\n";
             file << "      \"message\": \"" << issue.message << "\",\n";
             file << "      \"code\": \"" << issue.code_snippet << "\"\n";
             file << "    }";
             
             if (i < issues.size() - 1) {
                 file << ",";
             }
             
             file << "\n";
         }
         
         file << "  ]\n}\n";
         
         std::cout << "Results exported to " << filename << std::endl;
     }
 };
 
 void printUsage() {
     std::cout << "Usage: static_analyzer [options] <path>" << std::endl;
     std::cout << "Options:" << std::endl;
     std::cout << "  --help          Show this help message" << std::endl;
     std::cout << "  --json <file>   Export results to JSON file" << std::endl;
     std::cout << "  --verbose       Show detailed output" << std::endl;
 }
 
 int main(int argc, char* argv[]) {
     if (argc < 2) {
         printUsage();
         return 1;
     }
     
     std::string path;
     std::string json_output;
     bool verbose = false;
     
     // Parse command line arguments
     for (int i = 1; i < argc; ++i) {
         std::string arg = argv[i];
         
         if (arg == "--help") {
             printUsage();
             return 0;
         } else if (arg == "--json" && i + 1 < argc) {
             json_output = argv[++i];
         } else if (arg == "--verbose") {
             verbose = true;
         } else {
             path = arg;
         }
     }
     
     if (path.empty()) {
         std::cerr << "Error: No path specified" << std::endl;
         printUsage();
         return 1;
     }
     
     try {
         StaticAnalyzer analyzer;
         analyzer.registerRules();
         
         std::vector<std::string> files;
         
         // Check if path is a directory or a single file
         if (fs::is_directory(path)) {
             std::cout << "Scanning directory: " << path << std::endl;
             files = analyzer.findCppFiles(path);
             std::cout << "Found " << files.size() << " C++ files to analyze" << std::endl;
         } else {
             std::cout << "Analyzing file: " << path << std::endl;
             files.push_back(path);
         }
         
         if (files.empty()) {
             std::cout << "No C++ files found to analyze" << std::endl;
             return 0;
         }
         
         auto issues = analyzer.analyzeFiles(files);
         
         if (verbose) {
             analyzer.printIssues(issues);
         } else {
             std::cout << "Analysis completed. Found " << issues.size() << " issues." << std::endl;
         }
         
         if (!json_output.empty()) {
             analyzer.exportToJson(issues, json_output);
         }
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     return 0;
 }