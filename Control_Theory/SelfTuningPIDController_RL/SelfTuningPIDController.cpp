/**
 * @file SelfTuningPIDController.cpp
 * @brief Implementation of the SelfTuningPIDController class
 *
 * Orchestrates the two-phase PID gain optimization:
 *   1. SimpleTuning() — grid search over Kp/Ki/Kd combinations to find
 *      a reasonable starting point.
 *   2. Train() — reinforcement learning (Q-learning) to fine-tune gains
 *      by interacting with the process environment.
 *
 * After tuning, Run() executes the PID loop with fixed gains and returns
 * a time-series history for offline analysis.
 */

#include "SelfTuningPIDController.hpp"

SelfTuningPIDController::SelfTuningPIDController(std::shared_ptr<Environment> env,
                                                   std::shared_ptr<PIDController> pidController,
                                                   size_t hiddenSize)
    : env_(env), pidController_(pidController),
      rlAgent_(3, 3, hiddenSize), isTraining_(false) {}

/**
 * @brief Train PID gains via Q-learning episodes
 *
 * Each episode:
 *   1. Reset environment and PID state to a clean start.
 *   2. At every time step the RL agent proposes new [Kp, Ki, Kd] gains
 *      based on the current state [error, errorDeriv, lastControl].
 *   3. The PID controller computes a control signal with those gains.
 *   4. The environment advances one step and returns a reward based on
 *      how small the tracking error is.
 *   5. The agent updates its Q-network to prefer gains that produce
 *      higher cumulative reward.
 *
 * The best-performing gains (highest cumulative reward across all
 * episodes) are restored at the end of training.
 */
void SelfTuningPIDController::Train(int numEpisodes, int targetSyncFrequency,
                                      int maxStepsPerEpisode) {
    isTraining_.store(true);
    double highestReward = -std::numeric_limits<double>::infinity();
    std::vector<double> bestGains;

    for (int ep = 0; ep < numEpisodes && isTraining_.load(); ++ep) {
        // ── Start a fresh episode ──
        std::vector<double> state = env_->Reset();
        pidController_->Reset();

        if (auto* simpleEnv = dynamic_cast<SimpleProcessEnvironment*>(env_.get())) {
            simpleEnv->SetMaxSteps(maxStepsPerEpisode);
        }

        double cumulativeReward = 0.0;
        bool episodeFinished = false;

        for (int step = 0; step < maxStepsPerEpisode && !episodeFinished
                 && isTraining_.load(); ++step) {

            // Agent picks PID gains [Kp, Ki, Kd] from current state
            std::vector<double> proposedGains = rlAgent_.SelectAction(state);
            pidController_->UpdateParameters(
                proposedGains[0], proposedGains[1], proposedGains[2]);

            // PID computes a control signal from the current tracking error
            double trackingError  = env_->GetSetpoint() - env_->GetProcessValue();
            double controlSignal  = pidController_->Compute(trackingError);

            // Environment advances: returns next state, reward, and done flag
            auto [nextState, reward, done] = env_->Step(proposedGains, controlSignal);
            episodeFinished = done;

            // Q-network learns from this experience (skip step 0 — no prior state)
            if (step > 0) {
                rlAgent_.Learn(state, proposedGains, reward, nextState, done);
            }

            state = nextState;
            cumulativeReward += reward;
        }

        // Periodically copy Q-network weights to the target network
        // to stabilize the learning targets (standard DQN technique)
        if (ep % targetSyncFrequency == 0) {
            rlAgent_.SyncTargetNetwork();
        }

        // Track the best gains seen across all episodes
        if (cumulativeReward > highestReward) {
            highestReward = cumulativeReward;
            bestGains = pidController_->GetParameters();
            rlAgent_.SaveModel("best_pid_model.dat");
        }
    }

    // Restore the best gains found during training
    if (!bestGains.empty()) {
        pidController_->UpdateParameters(bestGains[0], bestGains[1], bestGains[2]);
    }
    isTraining_.store(false);
}

void SelfTuningPIDController::SimpleTuning(int numTests) {
    std::vector<double> bestParams = rlAgent_.GridSearch(env_, numTests);
    pidController_->UpdateParameters(bestParams[0], bestParams[1], bestParams[2]);
}

void SelfTuningPIDController::StopTraining() { isTraining_.store(false); }

std::vector<std::vector<double>> SelfTuningPIDController::Run(int numSteps, double setpoint) {
    std::vector<std::vector<double>> history;
    env_->Reset();
    pidController_->Reset();

    if (auto* simpleEnv = dynamic_cast<SimpleProcessEnvironment*>(env_.get())) {
        simpleEnv->SetSetpoint(setpoint);
        simpleEnv->SetMaxSteps(numSteps + 1);
    }

    auto currentGains = pidController_->GetParameters();

    for (int step = 0; step < numSteps; ++step) {
        double t = step * 0.01;
        double error = env_->GetSetpoint() - env_->GetProcessValue();
        double controlSignal = pidController_->Compute(error);
        env_->Step(currentGains, controlSignal);
        history.push_back({t, env_->GetSetpoint(), env_->GetProcessValue(), controlSignal});
    }
    return history;
}

void SelfTuningPIDController::SaveModel(const std::string& filename) {
    rlAgent_.SaveModel(filename);
}

void SelfTuningPIDController::LoadModel(const std::string& filename) {
    rlAgent_.LoadModel(filename);
}
