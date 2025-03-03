/**
 * @file RLAgent.cpp
 * @brief Implementation of the RLAgent class
 */

#include "RLAgent.hpp"

RLAgent::RLAgent(size_t stateSize, size_t actionSize, size_t hiddenSize,
                 double learningRate, double gamma,
                 double epsilon, double epsilonDecay, double epsilonMin)
    : stateSize_(stateSize), actionSize_(actionSize),
      qNetwork_(stateSize, hiddenSize, actionSize, learningRate),
      targetNetwork_(stateSize, hiddenSize, actionSize, learningRate),
      gamma_(gamma), epsilon_(epsilon), epsilonDecay_(epsilonDecay),
      epsilonMin_(epsilonMin), gen_(std::random_device{}()),
      actionRanges_({{0.0, 2.0}, {0.0, 1.0}, {0.0, 0.5}}) {
    SyncTargetNetwork();
}

std::vector<double> RLAgent::SelectAction(const std::vector<double>& state) {
    std::vector<double> action(actionSize_);

    std::uniform_real_distribution<double> epsDist(0.0, 1.0);
    bool isExploring = epsDist(gen_) < epsilon_;

    if (isExploring) {
        for (size_t i = 0; i < actionSize_; ++i) {
            std::uniform_real_distribution<double> actionDist(
                actionRanges_[i].first, actionRanges_[i].second);
            action[i] = actionDist(gen_);
        }
        if (actionSize_ >= 3) {
            action[0] = std::max({action[0], action[1], action[2]});
            action[2] = std::min({action[0], action[1], action[2]});
        }
    } else {
        std::vector<double> qValues = qNetwork_.Forward(state);
        for (size_t i = 0; i < actionSize_; ++i) {
            double normalizedQ = (qValues[i] + 1.0) / 2.0;
            double range = actionRanges_[i].second - actionRanges_[i].first;
            action[i] = actionRanges_[i].first + normalizedQ * range;
        }
    }
    return action;
}

double RLAgent::Learn(const std::vector<double>& state, const std::vector<double>& action,
                      double reward, const std::vector<double>& nextState, bool done) {
    // Validate inputs
    for (const auto& v : state) if (std::isnan(v) || std::isinf(v)) return 0.0;
    for (const auto& v : action) if (std::isnan(v) || std::isinf(v)) return 0.0;
    for (const auto& v : nextState) if (std::isnan(v) || std::isinf(v)) return 0.0;
    if (std::isnan(reward) || std::isinf(reward)) return 0.0;

    std::vector<double> nextQValues = targetNetwork_.Forward(nextState);
    std::vector<double> qValues = qNetwork_.Forward(state);
    std::vector<double> targetQValues = qValues;

    for (size_t i = 0; i < actionSize_; ++i) {
        targetQValues[i] = done ? reward : (reward + gamma_ * nextQValues[i]);
        targetQValues[i] = std::clamp(targetQValues[i], -1.0, 1.0);
    }

    double loss = qNetwork_.Train(state, targetQValues);
    epsilon_ = std::max(epsilonMin_, epsilon_ * epsilonDecay_);
    return loss;
}

void RLAgent::SyncTargetNetwork() {
    std::string tempFile = "temp_q_network.dat";
    qNetwork_.SaveModel(tempFile);
    targetNetwork_.LoadModel(tempFile);
    std::remove(tempFile.c_str());
}

void RLAgent::SaveModel(const std::string& filename) { qNetwork_.SaveModel(filename); }

void RLAgent::LoadModel(const std::string& filename) {
    qNetwork_.LoadModel(filename);
    targetNetwork_.LoadModel(filename);
}

double RLAgent::GetEpsilon() const { return epsilon_; }

std::vector<double> RLAgent::GridSearch(std::shared_ptr<Environment> env, int numTests) {
    std::vector<double> kpValues = {0.2, 0.5, 1.0, 1.5};
    std::vector<double> kiValues = {0.0, 0.1, 0.2, 0.5};
    std::vector<double> kdValues = {0.0, 0.02, 0.05, 0.1};

    double bestReward = -std::numeric_limits<double>::infinity();
    std::vector<double> bestParams = {0.5, 0.1, 0.05};
    auto pid = std::make_shared<PIDController>();

    for (double kp : kpValues) {
        for (double ki : kiValues) {
            for (double kd : kdValues) {
                pid->UpdateParameters(kp, ki, kd);
                pid->Reset();
                env->Reset();
                double totalReward = 0.0;

                for (int step = 0; step < numTests; ++step) {
                    double error = env->GetSetpoint() - env->GetProcessValue();
                    double control = pid->Compute(error);
                    auto [nextState, reward, done] = env->Step({kp, ki, kd}, control);
                    totalReward += reward;
                    if (done) break;
                }

                if (totalReward > bestReward) {
                    bestReward = totalReward;
                    bestParams = {kp, ki, kd};
                }
            }
        }
    }
    return bestParams;
}
