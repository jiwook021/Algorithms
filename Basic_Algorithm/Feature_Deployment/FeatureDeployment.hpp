/**
 * @file FeatureDeployment.hpp
 * @brief Feature deployment grouping algorithm
 * @details Given feature progress percentages and daily speeds, compute
 *          how many features are deployed together on each deployment day.
 *          Features must deploy in order; later features wait for earlier ones.
 */

#pragma once

#include <vector>

namespace FeatureDeployment {

/**
 * @brief Compute deployment groups.
 * @param progresses Current progress of each feature (0-99)
 * @param speeds     Daily work speed of each feature (>=1)
 * @return Vector where each element is the count of features deployed together
 */
std::vector<int> Solution(const std::vector<int>& progresses,
                          const std::vector<int>& speeds);

} // namespace FeatureDeployment
