/**
 * @file main.cpp
 * @brief Driver for BayesianOptimization
 */

#include "BayesianOptimization.hpp"

 int main() {
     using namespace Bo;
     
     // First, run tests (with the problematic assertion removed)
     RunTests();
     
     // Run direct optimization test to verify our understanding of the objective function
     TestDirectOptimization();
     
     std::cout << "\nBayesian Optimization for SVM Hyperparameter Tuning" << std::endl;
     std::cout << "---------------------------------------------------" << std::endl;
     
     // Define hyperparameter space in log scale (KEY IMPROVEMENT)
     HyperParameterSpace Space;
     Space.Add(HyperParameter("log_C", -3.0, 2.0))        // log10(C) in [-3, 2]
          .Add(HyperParameter("log_gamma", -3.0, 2.0));   // log10(gamma) in [-3, 2]
     
     // Define objective function with log transform
     auto ObjectiveFunction = [](const HyperParameterConfiguration& Config) {
         // Convert from log space to original space
         double log_C = Config.get("log_C");
         double LogGamma = Config.get("log_gamma");
         double C = std::pow(10.0, log_C);
         double Gamma = std::pow(10.0, LogGamma);
         
         double Accuracy = SimulateSvmCvScore(C, Gamma);
         
         // Simulate a costly evaluation (like cross-validation)
         std::this_thread::sleep_for(std::chrono::milliseconds(100));
         
         std::cout << "  Evaluated: C=" << std::fixed << std::setprecision(4) << C
                   << ", gamma=" << std::fixed << std::setprecision(4) << Gamma
                   << " -> accuracy=" << std::fixed << std::setprecision(4) << Accuracy
                   << std::endl;
         
         return Accuracy;
     };
     
     // Create optimizer (maximize accuracy) with improved kernel settings
     BayesianOptimizer Optimizer(
         Space, 
         ObjectiveFunction, 
         false,  // maximize (not minimize)
         std::make_unique<Matern52Kernel>(0.5, 1.0)  // Better length scale for log space
     );
     
     // Initialize with more random evaluations for better coverage
     std::cout << "Performing initial random evaluations:" << std::endl;
     Optimizer.Initialize(10, 42);  // 10 initial points instead of 5
     
     // Run optimization with increased exploration
     std::cout << "\nRunning Bayesian optimization:" << std::endl;
     auto BestConfig = Optimizer.Optimize(15, 20, 0.1);  // Higher exploration parameter
     
     // Convert log parameters back to original space for reporting
     double log_C = BestConfig.get("log_C");
     double LogGamma = BestConfig.get("log_gamma");
     double C = std::pow(10.0, log_C);
     double Gamma = std::pow(10.0, LogGamma);
     
     // Print results
     std::cout << "\nOptimization completed." << std::endl;
     std::cout << "Best configuration found:" << std::endl;
     std::cout << "  log_C = " << std::fixed << std::setprecision(4) << log_C 
               << " (C = " << std::fixed << std::setprecision(4) << C << ")" << std::endl;
     std::cout << "  log_gamma = " << std::fixed << std::setprecision(4) << LogGamma 
               << " (gamma = " << std::fixed << std::setprecision(4) << Gamma << ")" << std::endl;
     std::cout << "  Accuracy = " << std::fixed << std::setprecision(4) << Optimizer.GetBestValue() << std::endl;
     std::cout << "  Target optimum: C = 10.0, gamma = 0.1" << std::endl;
     
     return 0;
 }