/**
 * @file main.cpp
 * @brief Interactive driver for Principal Component Analysis.
 *
 * Demonstrates PCA on a practical scenario: analyzing student exam
 * scores across 4 subjects to discover hidden patterns of academic
 * ability, then projecting from 4D to 2D.
 */

#include "PrincipalComponentAnalysis.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using PCA = PrincipalComponentAnalysis<double>;

    std::cout << "=============================================\n"
              << "  Principal Component Analysis (PCA)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  PCA finds the most important patterns in data.\n"
              << "  If you have data with many features (columns), PCA\n"
              << "  discovers which features vary together and compresses\n"
              << "  them into a smaller set of new features called\n"
              << "  principal components.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Center the data (subtract the average of each\n"
              << "     feature so everything is relative to the mean).\n"
              << "  2. Find the direction where the data varies the most\n"
              << "     — this is the 1st principal component (PC1).\n"
              << "  3. Find the next direction of greatest variance that\n"
              << "     is perpendicular to PC1 — this is PC2.\n"
              << "  4. Continue until you have as many components as\n"
              << "     original features.\n"
              << "  5. Keep only the top few components to reduce\n"
              << "     dimensionality.\n\n"

              << "WHY PCA MATTERS IN MACHINE LEARNING\n"
              << "  - Dimensionality reduction: Turn 100 features into 10,\n"
              << "    making models faster to train without losing much info.\n"
              << "  - Noise removal: High-variance components carry signal;\n"
              << "    low-variance components are often just noise.\n"
              << "  - Visualization: Project high-dimensional data to 2D/3D\n"
              << "    so humans can see clusters and patterns.\n"
              << "  - Decorrelation: PCA outputs are uncorrelated, which\n"
              << "    helps algorithms that assume independent features.\n\n"

              << "WHAT THE OUTPUTS MEAN\n"
              << "  Principal component : A direction (vector) in the original\n"
              << "       feature space that captures variance.\n"
              << "  Explained variance  : The fraction of total variance\n"
              << "       captured by each component (all sum to 1.0).\n"
              << "       If PC1 explains 90%, one number captures 90% of\n"
              << "       the information in all the original features.\n"
              << "  Projection          : The data re-expressed using only\n"
              << "       the top components — fewer numbers per data point.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Student exam score analysis
    //
    // 10 students took 4 exams: Math, Physics, English, History.
    // Math and Physics scores are correlated (STEM ability).
    // English and History scores are correlated (Humanities ability).
    // PCA should discover these two underlying factors without being
    // told about them.
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Student Exam Score Analysis\n\n"
              << "  A school has 10 students who took 4 exams:\n"
              << "  Math, Physics, English, and History.\n\n"
              << "  The school suspects that 4 separate scores are\n"
              << "  redundant — Math and Physics tend to move together\n"
              << "  (STEM ability), and English and History move together\n"
              << "  (Humanities ability).  PCA will discover these hidden\n"
              << "  patterns and compress 4 scores into 2 numbers.\n\n";

    // 10 students, 4 subjects
    PCA::Matrix scores = {
        //  Math  Physics  English  History
        {  90,    85,      60,      55  },  // Student 0
        {  85,    80,      65,      60  },  // Student 1
        {  78,    75,      70,      68  },  // Student 2
        {  72,    70,      75,      72  },  // Student 3
        {  60,    55,      88,      85  },  // Student 4
        {  55,    50,      92,      90  },  // Student 5
        {  95,    92,      50,      48  },  // Student 6
        {  65,    62,      80,      78  },  // Student 7
        {  80,    78,      72,      70  },  // Student 8
        {  50,    48,      95,      92  },  // Student 9
    };

    std::vector<std::string> subjects = {"Math", "Physics", "English", "History"};
    std::vector<std::string> profiles = {
        "Strong STEM",          "Strong STEM",
        "Balanced",             "Balanced",
        "Strong Humanities",    "Strong Humanities",
        "Very Strong STEM",     "Slight Humanities",
        "Balanced-STEM",        "Very Strong Humanities",
    };

    // Print the dataset
    std::cout << "  Input Data: Exam Scores (" << scores.size()
              << " students, " << subjects.size() << " subjects)\n\n"
              << "    Student  "
              << std::setw(8) << "Math"
              << std::setw(9) << "Physics"
              << std::setw(9) << "English"
              << std::setw(9) << "History"
              << "   Profile\n"
              << "    -------  --------  --------  --------  --------"
              << "  ----------------------\n";

    for (std::size_t i = 0; i < scores.size(); ++i) {
        std::cout << "    " << std::setw(7) << i;
        for (std::size_t j = 0; j < scores[i].size(); ++j) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(0)
                      << scores[i][j];
        }
        std::cout << "  " << profiles[i] << "\n";
    }

    // Fit PCA
    std::cout << "\n  Running PCA on these " << scores[0].size()
              << " features...\n";

    PCA pca;
    pca.Fit(scores);

    // Results
    std::cout << "\n---------------------------------------------\n"
              << "  PCA Results\n"
              << "---------------------------------------------\n\n";

    // Show the mean
    const auto& mean = pca.GetMean();
    std::cout << "  Step 1 — Subtract the average of each subject:\n    ";
    for (std::size_t j = 0; j < subjects.size(); ++j) {
        std::cout << subjects[j] << "=" << std::setprecision(1) << mean[j];
        if (j + 1 < subjects.size()) std::cout << "  ";
    }
    std::cout << "\n\n";

    // Show principal components
    const auto& components = pca.GetComponents();
    const auto& variance = pca.GetExplainedVarianceRatio();

    std::cout << "  Step 2 — Principal components found:\n\n"
              << "  Each component is a direction vector showing how much\n"
              << "  each subject contributes.  Positive and negative signs\n"
              << "  that oppose each other reveal a contrast (e.g., STEM\n"
              << "  vs. Humanities).\n\n";

    for (std::size_t i = 0; i < components.size(); ++i) {
        std::cout << "    PC" << (i + 1) << " — explains "
                  << std::setprecision(1) << (variance[i] * 100.0)
                  << "% of variance\n"
                  << "      Weights: [";
        for (std::size_t j = 0; j < components[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setprecision(3) << std::showpos
                      << components[i][j] << std::noshowpos;
        }
        std::cout << "]\n"
                  << "               ";
        for (std::size_t j = 0; j < subjects.size(); ++j) {
            if (j > 0) std::cout << "  ";
            std::cout << std::setw(7) << subjects[j];
        }
        std::cout << "\n\n";
    }

    // Project to 2D
    double cum_var = (variance[0] + variance[1]) * 100.0;
    std::cout << "  Step 3 — Project from 4D to 2D (keep PC1 + PC2)\n\n"
              << "  These 2 components capture " << std::setprecision(1)
              << cum_var << "% of total variance.\n"
              << "  (The remaining " << std::setprecision(1)
              << (100.0 - cum_var)
              << "% is noise or minor variation — safe to drop.)\n\n";

    PCA::Matrix projected = pca.Transform(scores, 2);

    std::cout << "    Student     PC1       PC2     Profile\n"
              << "    -------  --------  --------  ----------------------\n";
    for (std::size_t i = 0; i < projected.size(); ++i) {
        std::cout << "    " << std::setw(7) << i
                  << "  " << std::setw(8) << std::setprecision(2)
                  << projected[i][0]
                  << "  " << std::setw(8) << projected[i][1]
                  << "  " << profiles[i] << "\n";
    }

    std::cout << "\n  Notice how students with similar profiles cluster\n"
              << "  together in the 2D projection.  PCA discovered the\n"
              << "  STEM/Humanities structure from raw scores alone —\n"
              << "  no labels were needed.\n\n"

              << "  In ML, this means a model trained on just 2 PCA\n"
              << "  features can be nearly as accurate as one using all 4\n"
              << "  original scores, but trains faster and generalizes\n"
              << "  better.\n";

    // Interactive section
    std::cout << "\n---------------------------------------------\n"
              << "  Enter a student's scores to project, or 'q' to quit.\n"
              << "  Enter: Math Physics English History\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  Scores (Math Physics English History): ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty() || input[0] == 'q' || input[0] == 'Q') {
            break;
        }

        double m, p, e, h;
        try {
            std::size_t pos = 0;
            m = std::stod(input, &pos);
            std::string rest = input.substr(pos);
            p = std::stod(rest, &pos);
            rest = rest.substr(pos);
            e = std::stod(rest, &pos);
            rest = rest.substr(pos);
            h = std::stod(rest);
        } catch (...) {
            std::cout << "    Invalid input. Example: 80 75 60 55\n";
            continue;
        }

        auto point_proj = pca.Transform({{m, p, e, h}}, 2);

        std::cout << "    Scores: Math=" << std::setprecision(0) << m
                  << "  Physics=" << p
                  << "  English=" << e
                  << "  History=" << h << "\n"
                  << "    Projected: PC1=" << std::setprecision(2)
                  << point_proj[0][0]
                  << "  PC2=" << point_proj[0][1] << "\n"
                  << "    (Compare with the table above to see which\n"
                  << "     students this one is closest to.)\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
