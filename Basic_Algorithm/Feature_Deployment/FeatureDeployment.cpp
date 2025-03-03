/**
 * @file FeatureDeployment.cpp
 * @brief Implementation of feature deployment algorithm
 */

#include "FeatureDeployment.hpp"

namespace FeatureDeployment {

std::vector<int> Solution(const std::vector<int>& progresses,
                          const std::vector<int>& speeds) {
    int n = static_cast<int>(progresses.size());
    if (n == 0) return {};

    // Calculate days to completion for each feature
    std::vector<int> days(n);
    for (int i = 0; i < n; ++i) {
        int remaining = 100 - progresses[i];
        days[i] = (remaining + speeds[i] - 1) / speeds[i];
    }

    // Group features by blocking front feature
    std::vector<int> answer;
    int frontDay = days[0];
    int groupCount = 1;

    for (int i = 1; i < n; ++i) {
        if (days[i] <= frontDay) {
            ++groupCount;
        } else {
            answer.push_back(groupCount);
            frontDay = days[i];
            groupCount = 1;
        }
    }
    answer.push_back(groupCount);

    return answer;
}

} // namespace FeatureDeployment
