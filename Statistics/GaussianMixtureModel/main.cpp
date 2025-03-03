/**
 * @file main.cpp
 * @brief Interactive driver for the Gaussian Mixture Model.
 *
 * Demonstrates GMM on a practical scenario: segmenting customers by
 * annual spending and visit frequency.  The program generates synthetic
 * customer data from 3 known segments, fits a GMM to recover those
 * segments without labels, and lets the user classify new customers.
 */

#include "GaussianMixtureModel.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
    using GMM = GaussianMixtureModel<double>;

    std::cout << "=============================================\n"
              << "  Gaussian Mixture Model (GMM)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  A GMM discovers hidden groups (clusters) in unlabeled data.\n"
              << "  Unlike K-Means, which assigns each point to exactly one\n"
              << "  cluster, a GMM gives a soft assignment — the probability\n"
              << "  that each point belongs to each cluster.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Assume the data came from K Gaussian (bell-curve)\n"
              << "     distributions, each with its own center (mean),\n"
              << "     spread (variance), and proportion (weight).\n"
              << "  2. The EM algorithm alternates two steps:\n"
              << "       E-step: For each point, estimate the probability\n"
              << "               that it came from each of the K clusters.\n"
              << "       M-step: Update each cluster's mean, variance, and\n"
              << "               weight using those probabilities.\n"
              << "  3. Repeat until the model stops improving (log-likelihood\n"
              << "     converges).\n\n"

              << "WHAT THE OUTPUTS MEAN\n"
              << "  Mean     : The center of a cluster in each dimension.\n"
              << "  Variance : How spread out the cluster is in each dimension.\n"
              << "  Weight   : The fraction of all data that belongs to this\n"
              << "             cluster (all weights sum to 1.0).\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Customer segmentation
    //
    // A store wants to understand its customers by looking at two
    // features that it already tracks:
    //   Dimension 0 : Annual spending ($1000s)
    //   Dimension 1 : Store visits per month
    //
    // There are 3 hidden customer segments:
    //   Segment A — Budget shoppers     : low spending, few visits
    //   Segment B — Regular shoppers    : moderate spending, moderate visits
    //   Segment C — Premium shoppers    : high spending, frequent visits
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Customer Segmentation\n\n"
              << "  A retail store has 300 customers.  For each customer it\n"
              << "  recorded two numbers:\n"
              << "    (1) Annual spending  (in $1000s)\n"
              << "    (2) Store visits per month\n\n"
              << "  The store suspects there are 3 types of customers but\n"
              << "  has no labels.  The GMM will discover those groups.\n\n";

    // Generate synthetic customer data
    std::mt19937 rng(42);
    std::vector<std::vector<double>> data;

    // Segment A — Budget shoppers: spend ~$2k, visit ~2x/month
    std::normal_distribution<double> a_spending(2.0, 0.8);
    std::normal_distribution<double> a_visits(2.0, 0.7);
    for (int i = 0; i < 100; ++i) {
        data.push_back({a_spending(rng), a_visits(rng)});
    }

    // Segment B — Regular shoppers: spend ~$6k, visit ~5x/month
    std::normal_distribution<double> b_spending(6.0, 1.0);
    std::normal_distribution<double> b_visits(5.0, 0.9);
    for (int i = 0; i < 100; ++i) {
        data.push_back({b_spending(rng), b_visits(rng)});
    }

    // Segment C — Premium shoppers: spend ~$12k, visit ~10x/month
    std::normal_distribution<double> c_spending(12.0, 1.2);
    std::normal_distribution<double> c_visits(10.0, 1.0);
    for (int i = 0; i < 100; ++i) {
        data.push_back({c_spending(rng), c_visits(rng)});
    }

    std::cout << "  True segments used to generate the data:\n"
              << "    Segment A (Budget)  : spending ~ $2k,  visits ~ 2/mo\n"
              << "    Segment B (Regular) : spending ~ $6k,  visits ~ 5/mo\n"
              << "    Segment C (Premium) : spending ~ $12k, visits ~ 10/mo\n\n"
              << "  Total customers: " << data.size() << "\n\n";

    // Fit a 3-component GMM
    GMM gmm(3, 200, 1e-6);
    gmm.Fit(data);

    // Display recovered components
    std::cout << "---------------------------------------------\n"
              << "  GMM Results — Discovered Segments\n"
              << "---------------------------------------------\n\n";

    const auto& components = gmm.GetComponents();
    for (std::size_t k = 0; k < components.size(); ++k) {
        std::cout << "  Segment " << k << ":\n"
                  << std::fixed << std::setprecision(1)
                  << "    Center         : (spending=$"
                  << components[k].mean[0] << "k, visits="
                  << components[k].mean[1] << "/mo)\n"
                  << "    Spread (stdev) : (spending=$"
                  << std::setprecision(2)
                  << std::sqrt(components[k].variance[0]) << "k, visits="
                  << std::sqrt(components[k].variance[1]) << "/mo)\n"
                  << "    Weight         : "
                  << std::setprecision(1)
                  << (components[k].weight * 100.0) << "% of customers\n\n";
    }

    // Log-likelihood
    double ll = gmm.LogLikelihood(data);
    std::cout << "  Log-likelihood: " << std::setprecision(2) << ll << "\n"
              << "  (Higher = model fits the data better.  This number is\n"
              << "   always negative.  Values closer to 0 mean better fit.)\n\n";

    // Show cluster assignments for a few sample customers
    auto labels = gmm.Predict(data);
    std::cout << "---------------------------------------------\n"
              << "  Sample Assignments\n"
              << "---------------------------------------------\n\n";

    for (int seg = 0; seg < 3; ++seg) {
        std::string seg_name = (seg == 0) ? "A (Budget)"
                             : (seg == 1) ? "B (Regular)"
                                          : "C (Premium)";
        std::cout << "  True segment " << seg_name << " -> GMM assigned:\n";
        int shown = 0;
        for (std::size_t i = static_cast<std::size_t>(seg) * 100;
             i < static_cast<std::size_t>(seg + 1) * 100 && shown < 5;
             ++i, ++shown) {
            std::cout << "    Customer ($"
                      << std::setprecision(1) << data[i][0]
                      << "k, " << data[i][1]
                      << " visits/mo)  -> Segment " << labels[i] << "\n";
        }
        std::cout << "\n";
    }

    // Show soft assignments for a few interesting points
    std::cout << "---------------------------------------------\n"
              << "  Soft Assignments (Responsibilities)\n"
              << "---------------------------------------------\n\n"
              << "  Unlike K-Means, GMM gives the probability of belonging\n"
              << "  to each cluster.  Points near a cluster center have\n"
              << "  ~100% probability for that cluster.  Points between\n"
              << "  clusters show split probabilities.\n\n";

    // Pick a few representative customers
    std::vector<std::vector<double>> sample_points = {
        {2.0, 2.0},    // clearly budget
        {6.0, 5.0},    // clearly regular
        {12.0, 10.0},  // clearly premium
        {4.0, 3.5},    // between budget and regular
    };

    auto resp = gmm.Responsibilities(sample_points);
    std::vector<std::string> descriptions = {
        "Budget-like  ($2k, 2 visits)",
        "Regular-like ($6k, 5 visits)",
        "Premium-like ($12k, 10 visits)",
        "Ambiguous    ($4k, 3.5 visits)",
    };

    for (std::size_t i = 0; i < sample_points.size(); ++i) {
        std::cout << "  " << descriptions[i] << ":\n    ";
        for (std::size_t k = 0; k < components.size(); ++k) {
            if (k > 0) std::cout << "  ";
            std::cout << "Seg " << k << "="
                      << std::setprecision(1) << (resp[i][k] * 100.0) << "%";
        }
        std::cout << "\n";
    }

    // Interactive classification
    std::cout << "\n---------------------------------------------\n"
              << "  Classify a new customer, or 'q' to quit.\n"
              << "  Enter: spending($k) visits(/mo)\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  Customer (spending visits): ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty() || input[0] == 'q' || input[0] == 'Q') {
            break;
        }

        double spending, visits;
        try {
            std::size_t pos1 = 0;
            spending = std::stod(input, &pos1);
            visits = std::stod(input.substr(pos1));
        } catch (...) {
            std::cout << "    Invalid input. Example: 5.0 4.0\n";
            continue;
        }

        auto point_resp = gmm.Responsibilities({{spending, visits}});
        auto point_label = gmm.Predict({{spending, visits}});

        std::cout << "    Customer ($" << std::setprecision(1)
                  << spending << "k, " << visits << " visits/mo)\n"
                  << "    Probabilities:\n";
        for (std::size_t k = 0; k < components.size(); ++k) {
            std::cout << "      Segment " << k << ": "
                      << std::setprecision(1) << (point_resp[0][k] * 100.0)
                      << "%\n";
        }
        std::cout << "    Assigned to: Segment " << point_label[0] << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
