/**
 * @file main.cpp
 * @brief Interactive driver for K-Means clustering.
 *
 * Demonstrates K-Means on a practical scenario: optimizing delivery
 * hub placement for a food delivery service.  The program clusters
 * restaurant locations into zones, finds optimal hub positions, and
 * lets the user assign new restaurants to zones interactively.
 */

#include "KMeansClustering.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using KM = KMeansClustering<double>;

    std::cout << "=============================================\n"
              << "  K-Means Clustering\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  K-Means divides data into K groups (clusters) so that\n"
              << "  points in the same group are close together.  Each\n"
              << "  cluster has a center called a centroid.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Pick K random starting centers (centroids).\n"
              << "  2. Assignment step: assign each point to the nearest\n"
              << "     centroid.\n"
              << "  3. Update step: move each centroid to the average\n"
              << "     position of the points assigned to it.\n"
              << "  4. Repeat steps 2-3 until the centroids stop moving\n"
              << "     (convergence).\n\n"

              << "WHAT THE OUTPUTS MEAN\n"
              << "  Centroid   : The center of a cluster — the average\n"
              << "               position of all points in that group.\n"
              << "  Inertia    : Total squared distance from each point\n"
              << "               to its centroid.  Lower = tighter clusters.\n"
              << "  Assignment : Which cluster each point belongs to.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Delivery hub placement
    //
    // A food delivery service has 30 restaurants across a city grid
    // (x = east-west blocks, y = north-south blocks).
    // It wants to set up 3 distribution hubs so that each hub is
    // as close as possible to the restaurants it serves.
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Delivery Hub Placement\n\n"
              << "  A food delivery service has 30 restaurants across a\n"
              << "  city.  It wants to set up 3 distribution hubs so that\n"
              << "  each hub is as close as possible to the restaurants\n"
              << "  it serves.\n\n"
              << "  K-Means will:\n"
              << "    - Group the 30 restaurants into 3 zones\n"
              << "    - Find the best location for each hub (centroid)\n"
              << "    - Assign each restaurant to its nearest hub\n\n";

    // 30 restaurants in 3 natural clusters
    KM::Matrix restaurants = {
        // Downtown cluster (around x=2, y=3)
        {1.5, 2.5}, {2.0, 3.0}, {2.5, 3.5}, {1.8, 2.8},
        {2.2, 2.6}, {1.7, 3.2}, {2.8, 3.0}, {2.0, 2.0},
        {1.5, 3.5}, {2.5, 2.5},

        // East-side cluster (around x=8, y=3)
        {7.5, 2.5}, {8.0, 3.0}, {8.5, 3.5}, {7.8, 2.8},
        {8.2, 2.6}, {7.7, 3.2}, {8.8, 3.0}, {8.0, 2.0},
        {7.5, 3.5}, {8.5, 2.5},

        // North cluster (around x=5, y=9)
        {4.5, 8.5}, {5.0, 9.0}, {5.5, 9.5}, {4.8, 8.8},
        {5.2, 8.6}, {4.7, 9.2}, {5.8, 9.0}, {5.0, 8.0},
        {4.5, 9.5}, {5.5, 8.5},
    };

    // Print the dataset
    std::cout << "  Restaurant Locations (" << restaurants.size()
              << " restaurants)\n\n";

    std::vector<std::string> area_names = {
        "Downtown area", "East-side area", "North area"};
    for (int area = 0; area < 3; ++area) {
        std::cout << "    " << area_names[static_cast<std::size_t>(area)]
                  << ":\n";
        for (int i = area * 10; i < (area + 1) * 10; ++i) {
            std::cout << "      Restaurant " << std::setw(2) << i
                      << ": (" << std::fixed << std::setprecision(1)
                      << restaurants[static_cast<std::size_t>(i)][0] << ", "
                      << restaurants[static_cast<std::size_t>(i)][1] << ")\n";
        }
    }

    // Fit K-Means with K=3
    std::cout << "\n  Running K-Means with K=3...\n";

    KM model(3, 200, 1e-6, 0);
    auto assignments = model.Fit(restaurants);

    // Display results
    std::cout << "\n---------------------------------------------\n"
              << "  K-Means Results — Delivery Zones\n"
              << "---------------------------------------------\n\n";

    const auto& centroids = model.GetCentroids();

    for (std::size_t k = 0; k < centroids.size(); ++k) {
        // Count restaurants in this zone
        int count = 0;
        for (int a : assignments) {
            if (a == static_cast<int>(k)) ++count;
        }

        std::cout << "  Zone " << k << ":\n"
                  << std::fixed << std::setprecision(1)
                  << "    Hub location (centroid): ("
                  << centroids[k][0] << ", " << centroids[k][1] << ")\n"
                  << "    Restaurants in zone    : " << count << "\n"
                  << "    Members:\n";

        for (std::size_t i = 0; i < restaurants.size(); ++i) {
            if (assignments[i] == static_cast<int>(k)) {
                double dist = KM::Distance(restaurants[i], centroids[k]);
                std::cout << "      (" << restaurants[i][0] << ", "
                          << restaurants[i][1] << ")  dist to hub="
                          << std::setprecision(2) << dist
                          << std::setprecision(1) << "\n";
            }
        }
        std::cout << "\n";
    }

    // Inertia
    double inertia = model.Inertia();
    std::cout << "  Inertia: " << std::setprecision(2) << inertia << "\n"
              << "  (Total squared distance from each restaurant to its hub.\n"
              << "   Lower means restaurants are closer to their hubs.)\n\n";

    // Interactive classification
    std::cout << "---------------------------------------------\n"
              << "  Assign a new restaurant to a zone, or 'q' to quit.\n"
              << "  Enter: x y\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  New restaurant (x y): ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty() || input[0] == 'q' || input[0] == 'Q') {
            break;
        }

        double x, y;
        try {
            std::size_t pos1 = 0;
            x = std::stod(input, &pos1);
            y = std::stod(input.substr(pos1));
        } catch (...) {
            std::cout << "    Invalid input. Example: 5.0 4.0\n";
            continue;
        }

        int zone = model.Predict({x, y});

        std::cout << "    Restaurant at (" << std::fixed << std::setprecision(1)
                  << x << ", " << y << ")\n"
                  << "    Distances to each hub:\n";
        for (std::size_t k = 0; k < centroids.size(); ++k) {
            double dist = KM::Distance({x, y}, centroids[k]);
            std::cout << "      Zone " << k << " hub ("
                      << centroids[k][0] << ", " << centroids[k][1]
                      << "): dist=" << std::setprecision(2) << dist;
            if (static_cast<int>(k) == zone) std::cout << "  <-- nearest";
            std::cout << std::setprecision(1) << "\n";
        }
        std::cout << "    Assigned to: Zone " << zone << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
