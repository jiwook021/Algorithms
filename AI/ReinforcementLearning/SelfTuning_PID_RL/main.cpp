/**
 * @file main.cpp
 * @brief Driver for SelfTuningPidRl -- grid-search demo.
 */
#include "SelfTuningPidRl.hpp"
#include <iostream>
#include <iomanip>

int main() {
    try {
        using namespace Rl;

        std::cout << "=== Self-Tuning PID Controller ===\n\n";

        auto env = std::make_shared<SimpleProcessEnvironment>(2.0, 1.0, 0.01, 0.01);
        auto pid = std::make_shared<PidController>(0.5, 0.1, 0.05, 0.01);
        SelfTuningPidController stc(env, pid, 32);

        // Grid search (fast, deterministic)
        std::cout << "Running grid search...\n";
        stc.SimpleTuning(200);

        auto params = pid->GetParameters();
        std::cout << "Best PID params: [" << std::fixed << std::setprecision(4)
                  << params[0] << ", " << params[1] << ", " << params[2] << "]\n\n";

        // Run test
        auto h = stc.Run(300, 1.0);
        std::cout << "Step response (every 50th):\n";
        for (size_t i = 0; i < h.size(); i += 50)
            std::cout << "  t=" << h[i][0] << "  sp=" << h[i][1]
                      << "  pv=" << h[i][2] << "  ctrl=" << h[i][3] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
