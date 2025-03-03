/**
 * @file main.cpp
 * @brief Driver for HealthPredictionNN
 */

#include "HealthPredictionNN.hpp"

int main(int argc, char* argv[]) {
    try {
        std::string Filename = (argc > 1) ? argv[1] : "health_data.csv";
        
        std::cout << "Loading data from " << Filename << "..." << std::endl;
        
        // Load data
        auto [Features, DataPair] = HealthDataLoader::LoadCsv(Filename);
        auto& [data, Targets] = DataPair;
        
        std::cout << "Loaded " << data.size() << " samples with " 
                  << Features.size() << " features." << std::endl;
        
        // Split data
        auto [TrainPair, TestPair] = HealthDataLoader::TrainTestSplit(data, Targets, 0.2);
        auto& [TrainData, TrainTargets] = TrainPair;
        auto& [TestData, TestTargets] = TestPair;
        
        std::cout << "Training set: " << TrainData.size() << " samples" << std::endl;
        std::cout << "Test set: " << TestData.size() << " samples" << std::endl;
        
        // Define network architecture
        std::vector<unsigned> HiddenLayers = {64, 32};
        
        // Create and train model
        HealthScorePredictor Predictor(
            Features, 
            HiddenLayers,
            HealthScorePredictor::ActivationType::RELU,
            HealthScorePredictor::ActivationType::SIGMOID,
            0.01,  // learning rate
            0.9    // momentum
        );
        
        std::cout << "Training model..." << std::endl;
        double ValidationError = Predictor.Train(
            TrainData, 
            TrainTargets,
            32,     // batch size
            1000,   // max epochs
            20,     // early stopping patience
            0.2     // validation split
        );
        
        std::cout << "Training completed with validation error: " << ValidationError << std::endl;
        
        // Evaluate model
        auto Metrics = HealthDataLoader::EvaluateModel(Predictor, TestData, TestTargets);
        
        std::cout << "Model evaluation:" << std::endl;
        for (const auto& [Metric, value] : Metrics) {
            std::cout << "  " << Metric << ": " << value << std::endl;
        }
        
        // Save model
        Predictor.SaveModel("health_model.dat");
        std::cout << "Model saved to health_model.dat" << std::endl;
        
        // Get feature importance
        auto Importance = Predictor.GetFeatureImportance();
        
        std::cout << "Feature importance:" << std::endl;
        for (const auto& [Feature, Score] : Importance) {
            std::cout << "  " << Feature << ": " << Score << std::endl;
        }
        
        // Example inferences with diverse, realistic profiles
std::cout << "\nHealth Score Predictions for Diverse Profiles:" << std::endl;
std::cout << "============================================" << std::endl;

// Define a set of diverse, realistic profiles for inference
std::vector<std::pair<std::string, std::vector<double>>> InferenceProfiles = {
    {"Young Healthy Adult", {
        28.0,    // age
        1.0,     // gender (1 = female)
        62000.0, // income
        16.0,    // education_years 
        8.0,     // sleep_hours
        6.0,     // physical_activity (days/week)
        8.0,     // diet_score (0-10)
        3.0,     // stress_level (0-10, lower is better)
        7.0,     // work_life_balance (0-10)
        22.5,    // bmi
        110.0,   // systolic_bp
        70.0,    // diastolic_bp
        165.0,   // cholesterol
        60.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.0,     // pollution_exposure (0-10)
        6.5,     // green_space_access (0-10)
        8.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        1.5,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        9.5,     // seat_belt_use (0-10)
        8.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        2.0,     // depression_score (0-10)
        2.5,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        1.0,     // family_history_risk (0-10)
        8.0,     // healthcare_access (0-10)
        7.0      // health_insurance_quality (0-10)
    }},
    {"Older Adult with Chronic Conditions", {
        68.0,    // age
        0.0,     // gender (0 = male)
        55000.0, // income
        14.0,    // education_years
        6.5,     // sleep_hours
        2.5,     // physical_activity (days/week)
        7.0,     // diet_score (0-10)
        4.5,     // stress_level (0-10, lower is better)
        6.0,     // work_life_balance (0-10)
        27.5,    // bmi
        138.0,   // systolic_bp
        85.0,    // diastolic_bp
        210.0,   // cholesterol
        74.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.5,     // pollution_exposure (0-10)
        5.0,     // green_space_access (0-10)
        5.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        1.0,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        9.0,     // seat_belt_use (0-10)
        7.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        3.0,     // depression_score (0-10)
        3.0,     // anxiety_score (0-10)
        2.5,     // chronic_diseases (count)
        4.0,     // family_history_risk (0-10)
        8.0,     // healthcare_access (0-10)
        7.0      // health_insurance_quality (0-10)
    }},
    {"Sedentary Smoker", {
        42.0,    // age
        0.0,     // gender (0 = male)
        48000.0, // income
        12.0,    // education_years
        5.5,     // sleep_hours
        1.0,     // physical_activity (days/week)
        4.0,     // diet_score (0-10)
        7.0,     // stress_level (0-10, lower is better)
        4.0,     // work_life_balance (0-10)
        31.0,    // bmi
        142.0,   // systolic_bp
        92.0,    // diastolic_bp
        225.0,   // cholesterol
        82.0,    // resting_heart_rate
        0.0,     // regular_checkups (0 = no)
        6.0,     // pollution_exposure (0-10)
        3.0,     // green_space_access (0-10)
        4.0,     // walkability_score (0-10)
        7.5,     // smoking (0-10)
        5.0,     // alcohol_consumption (drinks/week)
        0.5,     // recreational_drug_use (0-5)
        6.0,     // seat_belt_use (0-10)
        4.0,     // social_connections (0-10)
        2.0,     // community_engagement (0-10)
        5.0,     // depression_score (0-10)
        6.0,     // anxiety_score (0-10)
        1.0,     // chronic_diseases (count)
        3.0,     // family_history_risk (0-10)
        5.0,     // healthcare_access (0-10)
        4.0      // health_insurance_quality (0-10)
    }},
    {"Low-Income Student", {
        22.0,    // age
        1.0,     // gender (1 = female)
        18000.0, // income
        14.0,    // education_years (in progress)
        6.0,     // sleep_hours
        3.5,     // physical_activity (days/week)
        5.0,     // diet_score (0-10)
        8.0,     // stress_level (0-10, lower is better)
        5.0,     // work_life_balance (0-10)
        23.0,    // bmi
        118.0,   // systolic_bp
        75.0,    // diastolic_bp
        170.0,   // cholesterol
        72.0,    // resting_heart_rate
        0.0,     // regular_checkups (0 = no)
        4.0,     // pollution_exposure (0-10)
        5.0,     // green_space_access (0-10)
        7.0,     // walkability_score (0-10)
        2.0,     // smoking (0-10)
        3.0,     // alcohol_consumption (drinks/week)
        0.5,     // recreational_drug_use (0-5)
        8.0,     // seat_belt_use (0-10)
        7.0,     // social_connections (0-10)
        4.0,     // community_engagement (0-10)
        4.0,     // depression_score (0-10)
        5.0,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        2.0,     // family_history_risk (0-10)
        4.0,     // healthcare_access (0-10)
        3.0      // health_insurance_quality (0-10)
    }},
    {"Health-Conscious Professional", {
        38.0,    // age
        1.0,     // gender (1 = female)
        85000.0, // income
        18.0,    // education_years
        7.5,     // sleep_hours
        5.0,     // physical_activity (days/week)
        9.0,     // diet_score (0-10)
        4.0,     // stress_level (0-10, lower is better)
        7.0,     // work_life_balance (0-10)
        23.0,    // bmi
        116.0,   // systolic_bp
        74.0,    // diastolic_bp
        165.0,   // cholesterol
        64.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.0,     // pollution_exposure (0-10)
        7.0,     // green_space_access (0-10)
        7.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        2.0,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        10.0,    // seat_belt_use (0-10)
        8.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        2.0,     // depression_score (0-10)
        3.0,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        2.0,     // family_history_risk (0-10)
        9.0,     // healthcare_access (0-10)
        8.0      // health_insurance_quality (0-10)
    }}
};

// Make predictions for each profile
for (const auto& [ProfileName, ProfileValues] : InferenceProfiles) {
    std::cout << "\nProfile: " << ProfileName << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    // Display key health metrics
    std::cout << "Key metrics:" << std::endl;
    std::vector<std::pair<std::string, int>> KeyMetrics = {
        {"Age", 0}, {"BMI", 9}, {"Physical Activity", 5}, 
        {"Diet Score", 6}, {"Stress Level", 7}, {"Smoking", 18},
        {"Chronic Diseases", 26}
    };
    
    for (const auto& [Metric, Idx] : KeyMetrics) {
        std::cout << "  " << Metric << ": " << ProfileValues[Idx] << std::endl;
    }
    
    // Make prediction
    double HealthScore = Predictor.Predict(ProfileValues);
    std::cout << "\nPredicted health score: " << HealthScore << std::endl;
    
    // Explain prediction
    auto Explanations = Predictor.ExplainPrediction(ProfileValues);
    
    // Show top 3 positive contributors
    std::cout << "\nTop positive contributors:" << std::endl;
    int PosCount = 0;
    for (const auto& [Feature, Contribution] : Explanations) {
        if (Contribution > 0) {
            std::cout << "  " << Feature << ": +" << Contribution << std::endl;
            if (++PosCount >= 3) break;
        }
    }
    
    // Show top 3 negative contributors
    std::cout << "\nTop negative contributors:" << std::endl;
    int NegCount = 0;
    for (const auto& [Feature, Contribution] : Explanations) {
        if (Contribution < 0) {
            std::cout << "  " << Feature << ": " << Contribution << std::endl;
            if (++NegCount >= 3) break;
        }
    }
    
    std::cout << "------------------------------------------" << std::endl;
}
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}