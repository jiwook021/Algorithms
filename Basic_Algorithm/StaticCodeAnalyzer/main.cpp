/**
 * @file main.cpp
 * @brief Driver for StaticCodeAnalyzer
 */

#include "StaticCodeAnalyzer.hpp"

 int main(int argc, char* argv[]) {
     if (argc < 2) {
         PrintUsage();
         return 1;
     }
     
     std::string Path;
     std::string JsonOutput;
     bool Verbose = false;
     
     // Parse command line arguments
     for (int i = 1; i < argc; ++i) {
         std::string Arg = argv[i];
         
         if (Arg == "--help") {
             PrintUsage();
             return 0;
         } else if (Arg == "--json" && i + 1 < argc) {
             JsonOutput = argv[++i];
         } else if (Arg == "--verbose") {
             Verbose = true;
         } else {
             Path = Arg;
         }
     }
     
     if (Path.empty()) {
         std::cerr << "Error: No path specified" << std::endl;
         PrintUsage();
         return 1;
     }
     
     try {
         StaticAnalyzer Analyzer;
         Analyzer.RegisterRules();
         
         std::vector<std::string> Files;
         
         // Check if path is a directory or a single file
         if (Fs::is_directory(Path)) {
             std::cout << "Scanning directory: " << Path << std::endl;
             Files = Analyzer.FindCppFiles(Path);
             std::cout << "Found " << Files.size() << " C++ files to analyze" << std::endl;
         } else {
             std::cout << "Analyzing file: " << Path << std::endl;
             Files.push_back(Path);
         }
         
         if (Files.empty()) {
             std::cout << "No C++ files found to analyze" << std::endl;
             return 0;
         }
         
         auto Issues = Analyzer.AnalyzeFiles(Files);
         
         if (Verbose) {
             Analyzer.PrintIssues(Issues);
         } else {
             std::cout << "Analysis completed. Found " << Issues.size() << " issues." << std::endl;
         }
         
         if (!JsonOutput.empty()) {
             Analyzer.ExportToJson(Issues, JsonOutput);
         }
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     return 0;
 }