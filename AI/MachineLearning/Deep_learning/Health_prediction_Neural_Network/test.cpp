#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>

/**
 * @brief Test program for health score prediction
 * 
 * This program generates a diverse dataset with accurate health scores
 * and analyzes the impact of different factors on health outcomes.
 */
int main() {
    try {
        // Define output CSV file
        std::string filename = "health_data_accurate.csv";
        
        // Create output file
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        // Write CSV header
        file << "age,gender,income,education_years,sleep_hours,physical_activity,diet_score,"
             << "stress_level,work_life_balance,bmi,systolic_bp,diastolic_bp,cholesterol,"
             << "resting_heart_rate,regular_checkups,pollution_exposure,green_space_access,"
             << "walkability_score,smoking,alcohol_consumption,recreational_drug_use,"
             << "seat_belt_use,social_connections,community_engagement,depression_score,"
             << "anxiety_score,chronic_diseases,family_history_risk,healthcare_access,"
             << "health_insurance_quality,health_score" << std::endl;
        
        // Create random generator for diverse data
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Define health score calculation function
        auto calculate_health_score = [](const std::vector<double>& features) -> double {
            double score = 60.0; // Base score
            
            // Demographics impact
            score -= (features[0] - 30) * 0.2; // Age penalty
            score += (features[1] == 1) ? 1.0 : 0.0; // Gender bonus for female
            score += (features[2] - 40000) * 0.00005; // Income impact
            score += features[3] * 1.5; // Education impact
            
            // Lifestyle factors
            score += (features[4] - 7) * 2.5; // Sleep hours
            score += features[5] * 3.5; // Physical activity
            score += features[6] * 2.5; // Diet score
            score -= features[7] * 2.5; // Stress level
            score += features[8] * 2.0; // Work-life balance
            
            // Medical measurements
            double bmi = features[9];
            if (bmi < 18.5) {
                score -= (18.5 - bmi) * 2.0; // Underweight penalty
            } else if (bmi > 25) {
                score -= (bmi - 25) * 1.2; // Overweight penalty
            }
            
            score -= (features[10] - 120) * 0.15; // Systolic BP
            score -= (features[11] - 80) * 0.15; // Diastolic BP
            score -= (features[12] - 180) * 0.05; // Cholesterol
            score -= (features[13] - 70) * 0.1; // Resting heart rate
            score += features[14] * 4.0; // Regular checkups
            
            // Environmental factors
            score -= features[15] * 1.5; // Pollution exposure
            score += features[16] * 1.5; // Green space access
            score += features[17] * 1.0; // Walkability score
            
            // Behavioral factors
            score -= features[18] * 8.0; // Smoking
            score -= features[19] * 4.0; // Alcohol consumption
            score -= features[20] * 6.0; // Recreational drug use
            score += features[21] * 2.5; // Seat belt use
            
            // Social factors
            score += features[22] * 2.0; // Social connections
            score += features[23] * 1.0; // Community engagement
            
            // Mental health
            score -= features[24] * 3.0; // Depression score
            score -= features[25] * 2.5; // Anxiety score
            
            // Medical history
            score -= features[26] * 4.0; // Chronic diseases
            score -= features[27] * 2.0; // Family history risk
            
            // Healthcare access
            score += features[28] * 1.5; // Healthcare access
            score += features[29] * 1.0; // Health insurance quality
            
            // Clamp score to 0-100 range
            return std::max(0.0, std::min(100.0, score));
        };
        
        // Define predefined profiles for more realistic data
        std::vector<std::vector<double>> profile_templates = {
            // Young healthy profile
            {
                25.0, 1.0, 60000.0, 16.0, 8.0, 6.0, 8.0, 3.0, 7.0, 22.0, 110.0, 70.0, 160.0, 60.0,
                1.0, 2.0, 7.0, 8.0, 0.0, 1.0, 0.0, 10.0, 8.0, 6.0, 1.0, 2.0, 0.0, 1.0, 8.0, 7.0
            },
            // Middle-aged average health
            {
                45.0, 0.0, 55000.0, 14.0, 7.0, 3.0, 6.0, 5.0, 6.0, 26.0, 125.0, 82.0, 190.0, 72.0,
                1.0, 4.0, 5.0, 6.0, 1.0, 2.0, 0.0, 8.0, 6.0, 4.0, 3.0, 3.0, 1.0, 3.0, 7.0, 6.0
            },
            // Elderly with health issues
            {
                70.0, 0.0, 45000.0, 12.0, 6.0, 2.0, 6.0, 4.0, 5.0, 28.0, 140.0, 90.0, 210.0, 75.0,
                1.0, 3.0, 5.0, 4.0, 0.0, 1.0, 0.0, 9.0, 5.0, 5.0, 3.0, 4.0, 3.0, 4.0, 8.0, 7.0
            },
            // Young adult with poor habits
            {
                28.0, 0.0, 40000.0, 12.0, 6.0, 1.0, 4.0, 7.0, 4.0, 27.0, 120.0, 78.0, 185.0, 80.0,
                0.0, 5.0, 3.0, 5.0, 7.0, 5.0, 1.0, 7.0, 5.0, 3.0, 4.0, 5.0, 0.0, 2.0, 5.0, 4.0
            },
            // Health-conscious professional
            {
                38.0, 1.0, 80000.0, 18.0, 7.5, 5.0, 9.0, 4.0, 7.0, 23.0, 115.0, 75.0, 165.0, 65.0,
                1.0, 3.0, 7.0, 7.0, 0.0, 2.0, 0.0, 10.0, 8.0, 6.0, 2.0, 2.0, 0.0, 2.0, 9.0, 8.0
            },
            // Low-income individual
            {
                35.0, 0.0, 22000.0, 10.0, 6.0, 2.0, 5.0, 7.0, 5.0, 29.0, 130.0, 85.0, 200.0, 75.0,
                0.0, 6.0, 3.0, 3.0, 4.0, 3.0, 0.5, 7.0, 4.0, 3.0, 5.0, 6.0, 1.0, 3.0, 3.0, 2.0
            },
            // Fitness enthusiast
            {
                32.0, 1.0, 55000.0, 16.0, 8.0, 7.0, 9.0, 3.0, 7.0, 21.0, 105.0, 65.0, 150.0, 55.0,
                1.0, 3.0, 6.0, 7.0, 0.0, 1.0, 0.0, 10.0, 7.0, 5.0, 1.0, 2.0, 0.0, 1.0, 8.0, 7.0
            },
            // Obese individual
            {
                40.0, 0.0, 48000.0, 12.0, 6.0, 1.0, 4.0, 6.0, 5.0, 35.0, 145.0, 95.0, 240.0, 80.0,
                0.0, 4.0, 4.0, 4.0, 2.0, 3.0, 0.0, 8.0, 5.0, 4.0, 4.0, 5.0, 1.0, 3.0, 6.0, 5.0
            },
            // Student with stress
            {
                22.0, 1.0, 15000.0, 14.0, 6.0, 3.0, 5.0, 8.0, 4.0, 23.0, 115.0, 75.0, 170.0, 70.0,
                0.0, 4.0, 5.0, 7.0, 1.0, 3.0, 0.5, 9.0, 7.0, 4.0, 4.0, 6.0, 0.0, 1.0, 4.0, 3.0
            },
            // Retiree with good habits
            {
                68.0, 1.0, 45000.0, 14.0, 7.5, 4.0, 7.0, 3.0, 8.0, 24.0, 130.0, 80.0, 185.0, 68.0,
                1.0, 3.0, 6.0, 5.0, 0.0, 1.0, 0.0, 9.0, 7.0, 6.0, 2.0, 3.0, 2.0, 4.0, 8.0, 7.0
            }
        };
        
        std::cout << "Generating diverse health dataset with accurate scores..." << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        
        // Distributions for adding variation to templates
        std::normal_distribution<> age_var(0, 5);
        std::normal_distribution<> income_var(0, 5000);
        std::normal_distribution<> education_var(0, 1);
        std::normal_distribution<> sleep_var(0, 0.5);
        std::normal_distribution<> activity_var(0, 0.5);
        std::normal_distribution<> diet_var(0, 0.5);
        std::normal_distribution<> stress_var(0, 0.5);
        std::normal_distribution<> balance_var(0, 0.5);
        std::normal_distribution<> bmi_var(0, 1);
        std::normal_distribution<> bp_var(0, 5);
        std::normal_distribution<> cholesterol_var(0, 10);
        std::normal_distribution<> hr_var(0, 3);
        std::normal_distribution<> exposure_var(0, 1);
        std::normal_distribution<> green_var(0, 1);
        std::normal_distribution<> walk_var(0, 1);
        std::normal_distribution<> smoking_var(0, 0.5);
        std::normal_distribution<> alcohol_var(0, 0.5);
        std::normal_distribution<> drug_var(0, 0.1);
        std::normal_distribution<> seatbelt_var(0, 0.5);
        std::normal_distribution<> social_var(0, 1);
        std::normal_distribution<> community_var(0, 1);
        std::normal_distribution<> mental_var(0, 0.5);
        std::normal_distribution<> disease_var(0, 0.2);
        std::normal_distribution<> family_var(0, 0.5);
        std::normal_distribution<> healthcare_var(0, 0.5);
        
        // Create a distribution for profile template selection
        std::uniform_int_distribution<> template_dist(0, profile_templates.size() - 1);
        
        // Generate 50 diverse profiles based on the templates
        std::vector<std::vector<double>> profiles;
        
        for (int i = 0; i < 50; ++i) {
            // Select a random template
            int template_idx = template_dist(gen);
            std::vector<double> profile = profile_templates[template_idx];
            
            // Add variations to make unique profiles
            profile[0] = std::max(18.0, std::min(80.0, profile[0] + age_var(gen))); // Age
            profile[1] = profile[1]; // Keep gender as is
            profile[2] = std::max(15000.0, profile[2] + income_var(gen)); // Income
            profile[3] = std::max(8.0, std::min(20.0, profile[3] + education_var(gen))); // Education
            profile[4] = std::max(4.0, std::min(10.0, profile[4] + sleep_var(gen))); // Sleep
            profile[5] = std::max(0.0, std::min(7.0, profile[5] + activity_var(gen))); // Activity
            profile[6] = std::max(0.0, std::min(10.0, profile[6] + diet_var(gen))); // Diet
            profile[7] = std::max(0.0, std::min(10.0, profile[7] + stress_var(gen))); // Stress
            profile[8] = std::max(0.0, std::min(10.0, profile[8] + balance_var(gen))); // Work-life balance
            profile[9] = std::max(15.0, std::min(40.0, profile[9] + bmi_var(gen))); // BMI
            profile[10] = std::max(90.0, std::min(180.0, profile[10] + bp_var(gen))); // Systolic BP
            profile[11] = std::max(60.0, std::min(120.0, profile[11] + bp_var(gen))); // Diastolic BP
            profile[12] = std::max(120.0, std::min(300.0, profile[12] + cholesterol_var(gen))); // Cholesterol
            profile[13] = std::max(45.0, std::min(100.0, profile[13] + hr_var(gen))); // Heart rate
            profile[14] = profile[14]; // Keep checkups as is
            profile[15] = std::max(0.0, std::min(10.0, profile[15] + exposure_var(gen))); // Pollution
            profile[16] = std::max(0.0, std::min(10.0, profile[16] + green_var(gen))); // Green space
            profile[17] = std::max(0.0, std::min(10.0, profile[17] + walk_var(gen))); // Walkability
            profile[18] = std::max(0.0, std::min(10.0, profile[18] + smoking_var(gen))); // Smoking
            profile[19] = std::max(0.0, std::min(10.0, profile[19] + alcohol_var(gen))); // Alcohol
            profile[20] = std::max(0.0, std::min(5.0, profile[20] + drug_var(gen))); // Drugs
            profile[21] = std::max(0.0, std::min(10.0, profile[21] + seatbelt_var(gen))); // Seat belt
            profile[22] = std::max(0.0, std::min(10.0, profile[22] + social_var(gen))); // Social
            profile[23] = std::max(0.0, std::min(10.0, profile[23] + community_var(gen))); // Community
            profile[24] = std::max(0.0, std::min(10.0, profile[24] + mental_var(gen))); // Depression
            profile[25] = std::max(0.0, std::min(10.0, profile[25] + mental_var(gen))); // Anxiety
            profile[26] = std::max(0.0, std::min(5.0, profile[26] + disease_var(gen))); // Chronic diseases
            profile[27] = std::max(0.0, std::min(10.0, profile[27] + family_var(gen))); // Family history
            profile[28] = std::max(0.0, std::min(10.0, profile[28] + healthcare_var(gen))); // Healthcare access
            profile[29] = std::max(0.0, std::min(10.0, profile[29] + healthcare_var(gen))); // Insurance
            
            // Calculate accurate health score
            double health_score = calculate_health_score(profile);
            
            // Write to CSV
            for (const auto& feature : profile) {
                file << std::fixed << std::setprecision(2) << feature << ",";
            }
            file << std::fixed << std::setprecision(2) << health_score << std::endl;
            
            // Store profile for analysis
            profile.push_back(health_score);
            profiles.push_back(profile);
        }
        
        file.close();
        std::cout << "Generated 50 diverse profiles with accurate health scores" << std::endl;
        std::cout << "Saved to: " << filename << std::endl;
        
        // Analyze the generated dataset
        std::cout << "\nDataset Analysis:" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        
        // Calculate min, max, avg health score
        double min_score = 100.0;
        double max_score = 0.0;
        double sum_score = 0.0;
        
        for (const auto& profile : profiles) {
            double score = profile.back();
            min_score = std::min(min_score, score);
            max_score = std::max(max_score, score);
            sum_score += score;
        }
        
        double avg_score = sum_score / profiles.size();
        
        std::cout << "Health Score Statistics:" << std::endl;
        std::cout << "  Minimum: " << std::fixed << std::setprecision(1) << min_score << std::endl;
        std::cout << "  Maximum: " << std::fixed << std::setprecision(1) << max_score << std::endl;
        std::cout << "  Average: " << std::fixed << std::setprecision(1) << avg_score << std::endl;
        
        // Group scores by range
        std::vector<int> score_ranges(10, 0);
        for (const auto& profile : profiles) {
            int range_idx = std::min(9, static_cast<int>(profile.back() / 10.0));
            score_ranges[range_idx]++;
        }
        
        std::cout << "\nHealth Score Distribution:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  " << std::setw(2) << i*10 << "-" << std::setw(2) << (i+1)*10-1 << ": ";
            for (int j = 0; j < score_ranges[i]; ++j) {
                std::cout << "#";
            }
            std::cout << " (" << score_ranges[i] << ")" << std::endl;
        }
        
        // Display a few examples from different score ranges
        std::cout << "\nSample Profiles from Different Score Ranges:" << std::endl;
        
        // Sort profiles by health score
        std::sort(profiles.begin(), profiles.end(), 
                 [](const auto& a, const auto& b) { return a.back() < b.back(); });
        
        // Display profiles from different ranges
        std::vector<int> indices = {0, profiles.size()/4, profiles.size()/2, 3*profiles.size()/4, profiles.size()-1};
        
        for (int idx : indices) {
            const auto& profile = profiles[idx];
            double score = profile.back();
            
            std::cout << "\nProfile with health score " << std::fixed << std::setprecision(1) << score << ":" << std::endl;
            std::cout << "  Age: " << profile[0] << std::endl;
            std::cout << "  Gender: " << (profile[1] == 1.0 ? "Female" : "Male") << std::endl;
            std::cout << "  BMI: " << profile[9] << std::endl;
            std::cout << "  Physical Activity: " << profile[5] << "/7 days per week" << std::endl;
            std::cout << "  Diet Score: " << profile[6] << "/10" << std::endl;
            std::cout << "  Smoking: " << profile[18] << "/10" << std::endl;
            std::cout << "  Chronic Diseases: " << profile[26] << " conditions" << std::endl;
            std::cout << "  Stress Level: " << profile[7] << "/10" << std::endl;
            
            // Calculate major contributors to this score
            std::vector<std::pair<std::string, double>> contributions;
            std::vector<std::string> feature_names = {
                "age", "gender", "income", "education_years", "sleep_hours", "physical_activity", "diet_score",
                "stress_level", "work_life_balance", "bmi", "systolic_bp", "diastolic_bp", "cholesterol",
                "resting_heart_rate", "regular_checkups", "pollution_exposure", "green_space_access",
                "walkability_score", "smoking", "alcohol_consumption", "recreational_drug_use",
                "seat_belt_use", "social_connections", "community_engagement", "depression_score",
                "anxiety_score", "chronic_diseases", "family_history_risk", "healthcare_access",
                "health_insurance_quality"
            };
            
            // Calculate contribution of each feature
            contributions.emplace_back("age", -(profile[0] - 30) * 0.2);
            contributions.emplace_back("gender", (profile[1] == 1) ? 1.0 : 0.0);
            contributions.emplace_back("income", (profile[2] - 40000) * 0.00005);
            contributions.emplace_back("education_years", profile[3] * 1.5);
            contributions.emplace_back("sleep_hours", (profile[4] - 7) * 2.5);
            contributions.emplace_back("physical_activity", profile[5] * 3.5);
            contributions.emplace_back("diet_score", profile[6] * 2.5);
            contributions.emplace_back("stress_level", -profile[7] * 2.5);
            contributions.emplace_back("work_life_balance", profile[8] * 2.0);
            
            // BMI
            double bmi_contrib = 0.0;
            if (profile[9] < 18.5) {
                bmi_contrib = -(18.5 - profile[9]) * 2.0;
            } else if (profile[9] > 25) {
                bmi_contrib = -(profile[9] - 25) * 1.2;
            }
            contributions.emplace_back("bmi", bmi_contrib);
            
            contributions.emplace_back("systolic_bp", -(profile[10] - 120) * 0.15);
            contributions.emplace_back("diastolic_bp", -(profile[11] - 80) * 0.15);
            contributions.emplace_back("cholesterol", -(profile[12] - 180) * 0.05);
            contributions.emplace_back("resting_heart_rate", -(profile[13] - 70) * 0.1);
            contributions.emplace_back("regular_checkups", profile[14] * 4.0);
            contributions.emplace_back("pollution_exposure", -profile[15] * 1.5);
            contributions.emplace_back("green_space_access", profile[16] * 1.5);
            contributions.emplace_back("walkability_score", profile[17] * 1.0);
            contributions.emplace_back("smoking", -profile[18] * 8.0);
            contributions.emplace_back("alcohol_consumption", -profile[19] * 4.0);
            contributions.emplace_back("recreational_drug_use", -profile[20] * 6.0);
            contributions.emplace_back("seat_belt_use", profile[21] * 2.5);
            contributions.emplace_back("social_connections", profile[22] * 2.0);
            contributions.emplace_back("community_engagement", profile[23] * 1.0);
            contributions.emplace_back("depression_score", -profile[24] * 3.0);
            contributions.emplace_back("anxiety_score", -profile[25] * 2.5);
            contributions.emplace_back("chronic_diseases", -profile[26] * 4.0);
            contributions.emplace_back("family_history_risk", -profile[27] * 2.0);
            contributions.emplace_back("healthcare_access", profile[28] * 1.5);
            contributions.emplace_back("health_insurance_quality", profile[29] * 1.0);
            
            // Sort by absolute contribution
            std::sort(contributions.begin(), contributions.end(),
                     [](const auto& a, const auto& b) { 
                         return std::abs(a.second) > std::abs(b.second); 
                     });
            
            // Display top contributors
            std::cout << "  Top contributors:" << std::endl;
            for (int i = 0; i < 5; ++i) {
                const auto& [feature, contribution] = contributions[i];
                std::cout << "    " << feature << ": ";
                if (contribution > 0) std::cout << "+";
                std::cout << std::fixed << std::setprecision(1) << contribution << std::endl;
            }
        }
        
        std::cout << "\nNote: The generated dataset contains 50 profiles with health scores" << std::endl;
        std::cout << "ranging from " << std::fixed << std::setprecision(1) << min_score << " to " 
                  << max_score << " (average: " << avg_score << ")." << std::endl;
        std::cout << "Each score is accurately calculated based on the formula." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}