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
        std::string Filename = "health_data_accurate.csv";
        
        // Create output file
        std::ofstream File(Filename);
        if (!File.is_open()) {
            throw std::runtime_error("Failed to open file: " + Filename);
        }
        
        // Write CSV header
        File << "age,gender,income,education_years,sleep_hours,physical_activity,diet_score,"
             << "stress_level,work_life_balance,bmi,systolic_bp,diastolic_bp,cholesterol,"
             << "resting_heart_rate,regular_checkups,pollution_exposure,green_space_access,"
             << "walkability_score,smoking,alcohol_consumption,recreational_drug_use,"
             << "seat_belt_use,social_connections,community_engagement,depression_score,"
             << "anxiety_score,chronic_diseases,family_history_risk,healthcare_access,"
             << "health_insurance_quality,health_score" << std::endl;
        
        // Create random generator for diverse data
        std::random_device Rd;
        std::mt19937 Gen(Rd());
        
        // Define health score calculation function
        auto CalculateHealthScore = [](const std::vector<double>& Features) -> double {
            double Score = 60.0; // Base score
            
            // Demographics impact
            Score -= (Features[0] - 30) * 0.2; // Age penalty
            Score += (Features[1] == 1) ? 1.0 : 0.0; // Gender bonus for female
            Score += (Features[2] - 40000) * 0.00005; // Income impact
            Score += Features[3] * 1.5; // Education impact
            
            // Lifestyle factors
            Score += (Features[4] - 7) * 2.5; // sleep hours
            Score += Features[5] * 3.5; // Physical activity
            Score += Features[6] * 2.5; // Diet score
            Score -= Features[7] * 2.5; // Stress level
            Score += Features[8] * 2.0; // Work-life balance
            
            // Medical measurements
            double Bmi = Features[9];
            if (Bmi < 18.5) {
                Score -= (18.5 - Bmi) * 2.0; // Underweight penalty
            } else if (Bmi > 25) {
                Score -= (Bmi - 25) * 1.2; // Overweight penalty
            }
            
            Score -= (Features[10] - 120) * 0.15; // Systolic BP
            Score -= (Features[11] - 80) * 0.15; // Diastolic BP
            Score -= (Features[12] - 180) * 0.05; // Cholesterol
            Score -= (Features[13] - 70) * 0.1; // Resting heart rate
            Score += Features[14] * 4.0; // Regular checkups
            
            // Environmental factors
            Score -= Features[15] * 1.5; // Pollution exposure
            Score += Features[16] * 1.5; // Green space access
            Score += Features[17] * 1.0; // Walkability score
            
            // Behavioral factors
            Score -= Features[18] * 8.0; // Smoking
            Score -= Features[19] * 4.0; // Alcohol consumption
            Score -= Features[20] * 6.0; // Recreational drug use
            Score += Features[21] * 2.5; // Seat belt use
            
            // Social factors
            Score += Features[22] * 2.0; // Social connections
            Score += Features[23] * 1.0; // Community engagement
            
            // Mental health
            Score -= Features[24] * 3.0; // Depression score
            Score -= Features[25] * 2.5; // Anxiety score
            
            // Medical history
            Score -= Features[26] * 4.0; // Chronic diseases
            Score -= Features[27] * 2.0; // Family history risk
            
            // Healthcare access
            Score += Features[28] * 1.5; // Healthcare access
            Score += Features[29] * 1.0; // Health insurance quality
            
            // Clamp score to 0-100 range
            return std::max(0.0, std::min(100.0, Score));
        };
        
        // Define predefined profiles for more realistic data
        std::vector<std::vector<double>> ProfileTemplates = {
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
        std::normal_distribution<> AgeVar(0, 5);
        std::normal_distribution<> IncomeVar(0, 5000);
        std::normal_distribution<> EducationVar(0, 1);
        std::normal_distribution<> SleepVar(0, 0.5);
        std::normal_distribution<> ActivityVar(0, 0.5);
        std::normal_distribution<> DietVar(0, 0.5);
        std::normal_distribution<> StressVar(0, 0.5);
        std::normal_distribution<> BalanceVar(0, 0.5);
        std::normal_distribution<> BmiVar(0, 1);
        std::normal_distribution<> BpVar(0, 5);
        std::normal_distribution<> CholesterolVar(0, 10);
        std::normal_distribution<> HrVar(0, 3);
        std::normal_distribution<> ExposureVar(0, 1);
        std::normal_distribution<> GreenVar(0, 1);
        std::normal_distribution<> WalkVar(0, 1);
        std::normal_distribution<> SmokingVar(0, 0.5);
        std::normal_distribution<> AlcoholVar(0, 0.5);
        std::normal_distribution<> DrugVar(0, 0.1);
        std::normal_distribution<> SeatbeltVar(0, 0.5);
        std::normal_distribution<> SocialVar(0, 1);
        std::normal_distribution<> CommunityVar(0, 1);
        std::normal_distribution<> MentalVar(0, 0.5);
        std::normal_distribution<> DiseaseVar(0, 0.2);
        std::normal_distribution<> FamilyVar(0, 0.5);
        std::normal_distribution<> HealthcareVar(0, 0.5);
        
        // Create a distribution for profile template selection
        std::uniform_int_distribution<> TemplateDist(0, ProfileTemplates.size() - 1);
        
        // Generate 50 diverse profiles based on the templates
        std::vector<std::vector<double>> Profiles;
        
        for (int i = 0; i < 50; ++i) {
            // Select a random template
            int TemplateIdx = TemplateDist(Gen);
            std::vector<double> Profile = ProfileTemplates[TemplateIdx];
            
            // Add variations to make unique profiles
            Profile[0] = std::max(18.0, std::min(80.0, Profile[0] + AgeVar(Gen))); // Age
            Profile[1] = Profile[1]; // Keep gender as is
            Profile[2] = std::max(15000.0, Profile[2] + IncomeVar(Gen)); // Income
            Profile[3] = std::max(8.0, std::min(20.0, Profile[3] + EducationVar(Gen))); // Education
            Profile[4] = std::max(4.0, std::min(10.0, Profile[4] + SleepVar(Gen))); // sleep
            Profile[5] = std::max(0.0, std::min(7.0, Profile[5] + ActivityVar(Gen))); // Activity
            Profile[6] = std::max(0.0, std::min(10.0, Profile[6] + DietVar(Gen))); // Diet
            Profile[7] = std::max(0.0, std::min(10.0, Profile[7] + StressVar(Gen))); // Stress
            Profile[8] = std::max(0.0, std::min(10.0, Profile[8] + BalanceVar(Gen))); // Work-life balance
            Profile[9] = std::max(15.0, std::min(40.0, Profile[9] + BmiVar(Gen))); // BMI
            Profile[10] = std::max(90.0, std::min(180.0, Profile[10] + BpVar(Gen))); // Systolic BP
            Profile[11] = std::max(60.0, std::min(120.0, Profile[11] + BpVar(Gen))); // Diastolic BP
            Profile[12] = std::max(120.0, std::min(300.0, Profile[12] + CholesterolVar(Gen))); // Cholesterol
            Profile[13] = std::max(45.0, std::min(100.0, Profile[13] + HrVar(Gen))); // Heart rate
            Profile[14] = Profile[14]; // Keep checkups as is
            Profile[15] = std::max(0.0, std::min(10.0, Profile[15] + ExposureVar(Gen))); // Pollution
            Profile[16] = std::max(0.0, std::min(10.0, Profile[16] + GreenVar(Gen))); // Green space
            Profile[17] = std::max(0.0, std::min(10.0, Profile[17] + WalkVar(Gen))); // Walkability
            Profile[18] = std::max(0.0, std::min(10.0, Profile[18] + SmokingVar(Gen))); // Smoking
            Profile[19] = std::max(0.0, std::min(10.0, Profile[19] + AlcoholVar(Gen))); // Alcohol
            Profile[20] = std::max(0.0, std::min(5.0, Profile[20] + DrugVar(Gen))); // Drugs
            Profile[21] = std::max(0.0, std::min(10.0, Profile[21] + SeatbeltVar(Gen))); // Seat belt
            Profile[22] = std::max(0.0, std::min(10.0, Profile[22] + SocialVar(Gen))); // Social
            Profile[23] = std::max(0.0, std::min(10.0, Profile[23] + CommunityVar(Gen))); // Community
            Profile[24] = std::max(0.0, std::min(10.0, Profile[24] + MentalVar(Gen))); // Depression
            Profile[25] = std::max(0.0, std::min(10.0, Profile[25] + MentalVar(Gen))); // Anxiety
            Profile[26] = std::max(0.0, std::min(5.0, Profile[26] + DiseaseVar(Gen))); // Chronic diseases
            Profile[27] = std::max(0.0, std::min(10.0, Profile[27] + FamilyVar(Gen))); // Family history
            Profile[28] = std::max(0.0, std::min(10.0, Profile[28] + HealthcareVar(Gen))); // Healthcare access
            Profile[29] = std::max(0.0, std::min(10.0, Profile[29] + HealthcareVar(Gen))); // Insurance
            
            // Calculate accurate health score
            double HealthScore = CalculateHealthScore(Profile);
            
            // Write to CSV
            for (const auto& Feature : Profile) {
                File << std::fixed << std::setprecision(2) << Feature << ",";
            }
            File << std::fixed << std::setprecision(2) << HealthScore << std::endl;
            
            // Store profile for analysis
            Profile.push_back(HealthScore);
            Profiles.push_back(Profile);
        }
        
        File.close();
        std::cout << "Generated 50 diverse profiles with accurate health scores" << std::endl;
        std::cout << "Saved to: " << Filename << std::endl;
        
        // Analyze the generated dataset
        std::cout << "\nDataset Analysis:" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        
        // Calculate min, max, avg health score
        double MinScore = 100.0;
        double MaxScore = 0.0;
        double SumScore = 0.0;
        
        for (const auto& Profile : Profiles) {
            double Score = Profile.back();
            MinScore = std::min(MinScore, Score);
            MaxScore = std::max(MaxScore, Score);
            SumScore += Score;
        }
        
        double AvgScore = SumScore / Profiles.size();
        
        std::cout << "Health Score Statistics:" << std::endl;
        std::cout << "  Minimum: " << std::fixed << std::setprecision(1) << MinScore << std::endl;
        std::cout << "  Maximum: " << std::fixed << std::setprecision(1) << MaxScore << std::endl;
        std::cout << "  Average: " << std::fixed << std::setprecision(1) << AvgScore << std::endl;
        
        // Group scores by range
        std::vector<int> ScoreRanges(10, 0);
        for (const auto& Profile : Profiles) {
            int RangeIdx = std::min(9, static_cast<int>(Profile.back() / 10.0));
            ScoreRanges[RangeIdx]++;
        }
        
        std::cout << "\nHealth Score Distribution:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  " << std::setw(2) << i*10 << "-" << std::setw(2) << (i+1)*10-1 << ": ";
            for (int j = 0; j < ScoreRanges[i]; ++j) {
                std::cout << "#";
            }
            std::cout << " (" << ScoreRanges[i] << ")" << std::endl;
        }
        
        // Display a few examples from different score ranges
        std::cout << "\nSample Profiles from Different Score Ranges:" << std::endl;
        
        // Sort profiles by health score
        std::sort(Profiles.begin(), Profiles.end(), 
                 [](const auto& a, const auto& b) { return a.back() < b.back(); });
        
        // Display profiles from different ranges
        std::vector<int> Indices = {0, Profiles.size()/4, Profiles.size()/2, 3*Profiles.size()/4, Profiles.size()-1};
        
        for (int Idx : Indices) {
            const auto& Profile = Profiles[Idx];
            double Score = Profile.back();
            
            std::cout << "\nProfile with health score " << std::fixed << std::setprecision(1) << Score << ":" << std::endl;
            std::cout << "  Age: " << Profile[0] << std::endl;
            std::cout << "  Gender: " << (Profile[1] == 1.0 ? "Female" : "Male") << std::endl;
            std::cout << "  BMI: " << Profile[9] << std::endl;
            std::cout << "  Physical Activity: " << Profile[5] << "/7 days per week" << std::endl;
            std::cout << "  Diet Score: " << Profile[6] << "/10" << std::endl;
            std::cout << "  Smoking: " << Profile[18] << "/10" << std::endl;
            std::cout << "  Chronic Diseases: " << Profile[26] << " conditions" << std::endl;
            std::cout << "  Stress Level: " << Profile[7] << "/10" << std::endl;
            
            // Calculate major contributors to this score
            std::vector<std::pair<std::string, double>> Contributions;
            std::vector<std::string> FeatureNames = {
                "age", "gender", "income", "education_years", "sleep_hours", "physical_activity", "diet_score",
                "stress_level", "work_life_balance", "bmi", "systolic_bp", "diastolic_bp", "cholesterol",
                "resting_heart_rate", "regular_checkups", "pollution_exposure", "green_space_access",
                "walkability_score", "smoking", "alcohol_consumption", "recreational_drug_use",
                "seat_belt_use", "social_connections", "community_engagement", "depression_score",
                "anxiety_score", "chronic_diseases", "family_history_risk", "healthcare_access",
                "health_insurance_quality"
            };
            
            // Calculate contribution of each feature
            Contributions.emplace_back("age", -(Profile[0] - 30) * 0.2);
            Contributions.emplace_back("gender", (Profile[1] == 1) ? 1.0 : 0.0);
            Contributions.emplace_back("income", (Profile[2] - 40000) * 0.00005);
            Contributions.emplace_back("education_years", Profile[3] * 1.5);
            Contributions.emplace_back("sleep_hours", (Profile[4] - 7) * 2.5);
            Contributions.emplace_back("physical_activity", Profile[5] * 3.5);
            Contributions.emplace_back("diet_score", Profile[6] * 2.5);
            Contributions.emplace_back("stress_level", -Profile[7] * 2.5);
            Contributions.emplace_back("work_life_balance", Profile[8] * 2.0);
            
            // BMI
            double BmiContrib = 0.0;
            if (Profile[9] < 18.5) {
                BmiContrib = -(18.5 - Profile[9]) * 2.0;
            } else if (Profile[9] > 25) {
                BmiContrib = -(Profile[9] - 25) * 1.2;
            }
            Contributions.emplace_back("bmi", BmiContrib);
            
            Contributions.emplace_back("systolic_bp", -(Profile[10] - 120) * 0.15);
            Contributions.emplace_back("diastolic_bp", -(Profile[11] - 80) * 0.15);
            Contributions.emplace_back("cholesterol", -(Profile[12] - 180) * 0.05);
            Contributions.emplace_back("resting_heart_rate", -(Profile[13] - 70) * 0.1);
            Contributions.emplace_back("regular_checkups", Profile[14] * 4.0);
            Contributions.emplace_back("pollution_exposure", -Profile[15] * 1.5);
            Contributions.emplace_back("green_space_access", Profile[16] * 1.5);
            Contributions.emplace_back("walkability_score", Profile[17] * 1.0);
            Contributions.emplace_back("smoking", -Profile[18] * 8.0);
            Contributions.emplace_back("alcohol_consumption", -Profile[19] * 4.0);
            Contributions.emplace_back("recreational_drug_use", -Profile[20] * 6.0);
            Contributions.emplace_back("seat_belt_use", Profile[21] * 2.5);
            Contributions.emplace_back("social_connections", Profile[22] * 2.0);
            Contributions.emplace_back("community_engagement", Profile[23] * 1.0);
            Contributions.emplace_back("depression_score", -Profile[24] * 3.0);
            Contributions.emplace_back("anxiety_score", -Profile[25] * 2.5);
            Contributions.emplace_back("chronic_diseases", -Profile[26] * 4.0);
            Contributions.emplace_back("family_history_risk", -Profile[27] * 2.0);
            Contributions.emplace_back("healthcare_access", Profile[28] * 1.5);
            Contributions.emplace_back("health_insurance_quality", Profile[29] * 1.0);
            
            // Sort by absolute contribution
            std::sort(Contributions.begin(), Contributions.end(),
                     [](const auto& a, const auto& b) { 
                         return std::abs(a.second) > std::abs(b.second); 
                     });
            
            // Display top contributors
            std::cout << "  Top contributors:" << std::endl;
            for (int i = 0; i < 5; ++i) {
                const auto& [Feature, Contribution] = Contributions[i];
                std::cout << "    " << Feature << ": ";
                if (Contribution > 0) std::cout << "+";
                std::cout << std::fixed << std::setprecision(1) << Contribution << std::endl;
            }
        }
        
        std::cout << "\nNote: The generated dataset contains 50 profiles with health scores" << std::endl;
        std::cout << "ranging from " << std::fixed << std::setprecision(1) << MinScore << " to " 
                  << MaxScore << " (average: " << AvgScore << ")." << std::endl;
        std::cout << "Each score is accurately calculated based on the formula." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}