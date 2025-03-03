/**
 * @file main.cpp
 * @brief Interactive driver for RL-based self-tuning PID controller
 * @details Explains the self-tuning process, runs grid search + RL training,
 *          and evaluates the tuned controller on step-response tests.
 */

#include "SelfTuningPIDController.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

int main() {
    try {
        std::cout << "=============================================\n"
                  << "  Self-Tuning PID Controller (RL-Based)\n"
                  << "=============================================\n\n"

                  << "PURPOSE\n"
                  << "  A standard PID controller requires hand-tuning three gains\n"
                  << "  (Kp, Ki, Kd) for each new process.  This program automates\n"
                  << "  that tuning by combining two optimization strategies:\n"
                  << "    1. Grid Search  — exhaustive sweep over gain combinations\n"
                  << "    2. Q-Learning   — a reinforcement learning agent that\n"
                  << "       interacts with the process and learns which gains\n"
                  << "       minimize tracking error over time.\n\n"

                  << "HOW IT WORKS\n"
                  << "  The RL agent observes the system state:\n"
                  << "    state = [error, error_derivative, last_control_signal]\n"
                  << "  and proposes PID gains [Kp, Ki, Kd] as its action.\n"
                  << "  The PID controller uses those gains to compute a control\n"
                  << "  signal that drives the process.  After each step, the\n"
                  << "  agent receives a reward:\n"
                  << "    reward = -error^2 - 0.01*control^2 + 0.5*(if |error|<0.1)\n"
                  << "  which penalizes large errors and excessive control effort\n"
                  << "  while rewarding tight tracking.\n\n"

                  << "  The agent's neural network (Q-network) learns to map\n"
                  << "  states to Q-values that predict cumulative future reward.\n"
                  << "  Over many episodes, it discovers gains that balance fast\n"
                  << "  response with stability.\n\n"

                  << "PROCESS MODEL\n"
                  << "  The simulated first-order process follows:\n"
                  << "    dPV/dt = (gain * control - PV) / tau\n"
                  << "  where PV is the process value, tau is the time constant,\n"
                  << "  and gain is the system gain.  Optional Gaussian noise\n"
                  << "  simulates real-world disturbances.\n\n"

                  << "TUNING PIPELINE\n"
                  << "  Phase 1: Grid Search — tests Kp x Ki x Kd combinations\n"
                  << "           to find the best starting point.\n"
                  << "  Phase 2: RL Training — the Q-learning agent fine-tunes\n"
                  << "           gains by running multiple episodes.\n"
                  << "  Phase 3: Evaluation  — runs the tuned PID on step-response\n"
                  << "           tests and reports tracking quality (MSE).\n"
                  << "---------------------------------------------\n\n";

        // --- Environment setup ---
        double timeConstant = 2.0;   // process response speed
        double processGain  = 1.0;   // steady-state gain
        double dt           = 0.01;  // 100 Hz simulation rate
        double noise        = 0.01;  // 1% disturbance noise

        auto env = std::make_shared<SimpleProcessEnvironment>(
            timeConstant, processGain, dt, noise);

        std::cout << "Process Configuration\n"
                  << "  Time constant (tau) : " << timeConstant
                  << " s  (larger = slower response)\n"
                  << "  System gain         : " << processGain << "\n"
                  << "  Simulation rate     : " << (1.0 / dt) << " Hz\n"
                  << "  Noise std-dev       : " << noise << "\n\n";

        // --- Initial PID gains (conservative starting point) ---
        double initKp = 0.5, initKi = 0.1, initKd = 0.05;
        auto pid = std::make_shared<PIDController>(initKp, initKi, initKd, dt);

        std::cout << "Initial PID Gains (before tuning)\n"
                  << "  Kp = " << initKp
                  << "  (proportional — reacts to current error)\n"
                  << "  Ki = " << initKi
                  << "  (integral — eliminates steady-state offset)\n"
                  << "  Kd = " << initKd
                  << "  (derivative — dampens oscillations)\n\n";

        // --- Create self-tuning system ---
        size_t hiddenNeurons = 32;
        SelfTuningPIDController selfTuningPid(env, pid, hiddenNeurons);

        std::cout << "Q-Network Architecture\n"
                  << "  Input  : 3 neurons  [error, error_deriv, last_control]\n"
                  << "  Hidden : " << hiddenNeurons << " neurons (ReLU activation)\n"
                  << "  Output : 3 neurons  [Q_Kp, Q_Ki, Q_Kd]\n\n";

        // ================================================================
        // Phase 1: Grid Search
        // ================================================================
        int gridSearchSteps = 200;
        std::cout << "---------------------------------------------\n"
                  << "  Phase 1: Grid Search\n"
                  << "---------------------------------------------\n"
                  << "  Testing Kp x Ki x Kd combinations:\n"
                  << "    Kp: {0.2, 0.5, 1.0, 1.5}\n"
                  << "    Ki: {0.0, 0.1, 0.2, 0.5}\n"
                  << "    Kd: {0.0, 0.02, 0.05, 0.1}\n"
                  << "  Total: 4 x 4 x 4 = 64 combinations\n"
                  << "  Each tested for " << gridSearchSteps << " steps\n"
                  << "  Evaluating..." << std::flush;

        selfTuningPid.SimpleTuning(gridSearchSteps);

        auto gridParams = pid->GetParameters();
        std::cout << " done.\n\n"
                  << "  Best grid-search gains:\n"
                  << std::fixed << std::setprecision(4)
                  << "    Kp = " << gridParams[0] << "\n"
                  << "    Ki = " << gridParams[1] << "\n"
                  << "    Kd = " << gridParams[2] << "\n\n";

        // ================================================================
        // Phase 2: RL Training
        // ================================================================
        int numEpisodes        = 10;
        int targetSyncInterval = 5;
        int stepsPerEpisode    = 100;

        std::cout << "---------------------------------------------\n"
                  << "  Phase 2: RL Fine-Tuning (Q-Learning)\n"
                  << "---------------------------------------------\n"
                  << "  Episodes           : " << numEpisodes << "\n"
                  << "  Steps per episode  : " << stepsPerEpisode << "\n"
                  << "  Target net sync    : every " << targetSyncInterval << " episodes\n"
                  << "  Exploration        : epsilon-greedy (1.0 -> 0.01)\n"
                  << "  Discount factor    : 0.99\n"
                  << "  Training..." << std::flush;

        selfTuningPid.Train(numEpisodes, targetSyncInterval, stepsPerEpisode);

        auto rlParams = pid->GetParameters();
        std::cout << " done.\n\n"
                  << "  Post-RL gains:\n"
                  << "    Kp = " << rlParams[0] << "\n"
                  << "    Ki = " << rlParams[1] << "\n"
                  << "    Kd = " << rlParams[2] << "\n\n";

        std::cout << "  Gain comparison:\n"
                  << "    " << std::setw(6) << "Gain"
                  << std::setw(12) << "Initial"
                  << std::setw(12) << "Grid"
                  << std::setw(12) << "RL" << "\n"
                  << "    " << std::setw(6) << "Kp"
                  << std::setw(12) << initKp
                  << std::setw(12) << gridParams[0]
                  << std::setw(12) << rlParams[0] << "\n"
                  << "    " << std::setw(6) << "Ki"
                  << std::setw(12) << initKi
                  << std::setw(12) << gridParams[1]
                  << std::setw(12) << rlParams[1] << "\n"
                  << "    " << std::setw(6) << "Kd"
                  << std::setw(12) << initKd
                  << std::setw(12) << gridParams[2]
                  << std::setw(12) << rlParams[2] << "\n\n";

        // ================================================================
        // Phase 3: Evaluation
        // ================================================================
        std::cout << "---------------------------------------------\n"
                  << "  Phase 3: Step-Response Evaluation\n"
                  << "---------------------------------------------\n";

        int evalSteps = 300;
        double setpoints[] = {1.0, 0.5};

        for (double sp : setpoints) {
            auto history = selfTuningPid.Run(evalSteps, sp);

            if (!history.empty()) {
                double mse = 0.0;
                double maxError = 0.0;
                for (const auto& pt : history) {
                    double err = pt[1] - pt[2];
                    mse += err * err;
                    maxError = std::max(maxError, std::abs(err));
                }
                mse /= history.size();

                double finalPV   = history.back()[2];
                double finalErr  = std::abs(sp - finalPV);

                std::cout << "  Setpoint = " << sp << " V\n"
                          << "    MSE               : " << mse
                          << "  (mean squared tracking error)\n"
                          << "    Max error          : " << maxError
                          << "  (worst-case deviation)\n"
                          << "    Final value        : " << finalPV << "\n"
                          << "    Steady-state error : " << finalErr << "\n\n";
            }
        }

        // Save the trained model
        selfTuningPid.SaveModel("trained_pid_model.dat");
        std::cout << "  Model saved to trained_pid_model.dat\n\n"
                  << "Simulation completed successfully.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
