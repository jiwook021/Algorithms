/**
 * @file test_SelfTuningPidRl.cpp
 * @brief Unit tests for the SelfTuningPidRl module.
 */
#include <gtest/gtest.h>
#include "SelfTuningPidRl.hpp"
#include <cmath>

using namespace Rl;

// ======================== PidController ========================

TEST(PidController, ProportionalOnly) {
    PidController pid(1.0, 0.0, 0.0, 0.01);
    double out = pid.Compute(1.0);
    EXPECT_NEAR(out, 1.0, 1e-6);
}

TEST(PidController, Reset) {
    PidController pid(1.0, 1.0, 1.0, 0.01);
    pid.Compute(5.0);
    pid.Reset();
    auto params = pid.GetParameters();
    EXPECT_EQ(params.size(), 3u);
}

TEST(PidController, UpdateParameters) {
    PidController pid;
    pid.UpdateParameters(2.0, 0.5, 0.1);
    auto p = pid.GetParameters();
    EXPECT_DOUBLE_EQ(p[0], 2.0);
    EXPECT_DOUBLE_EQ(p[1], 0.5);
    EXPECT_DOUBLE_EQ(p[2], 0.1);
}

TEST(PidController, OutputIsBounded) {
    PidController pid(100.0, 0.0, 0.0, 0.01);
    double out = pid.Compute(100.0);
    EXPECT_LE(out, 10.0);
    EXPECT_GE(out, -10.0);
}

// ======================== NeuralNetwork ========================

TEST(NeuralNetwork, ForwardOutputSize) {
    NeuralNetwork nn(3, 8, 3);
    auto out = nn.Forward({1.0, 0.5, -0.5});
    EXPECT_EQ(out.size(), 3u);
}

TEST(NeuralNetwork, ForwardOutputBounded) {
    NeuralNetwork nn(2, 4, 2);
    auto out = nn.Forward({5.0, -5.0});
    for (double v : out) {
        EXPECT_GE(v, -1.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST(NeuralNetwork, TrainReducesLoss) {
    NeuralNetwork nn(2, 4, 1, 0.01);
    double loss1 = nn.Train({1.0, 0.0}, {0.5});
    for (int i = 0; i < 100; ++i) nn.Train({1.0, 0.0}, {0.5});
    double loss2 = nn.Train({1.0, 0.0}, {0.5});
    EXPECT_LT(loss2, loss1 + 0.01);
}

// ======================== SimpleProcessEnvironment ========================

TEST(SimpleProcessEnvironment, ResetReturnsState) {
    SimpleProcessEnvironment env;
    auto state = env.Reset();
    EXPECT_EQ(state.size(), 3u);
    EXPECT_NEAR(state[0], 1.0, 1e-6);  // error = setpoint(1) - pv(0)
}

TEST(SimpleProcessEnvironment, StepReturnsCorrectTuple) {
    SimpleProcessEnvironment env;
    env.Reset();
    auto [state, reward, done] = env.Step({1.0, 0.1, 0.05}, 1.0);
    EXPECT_EQ(state.size(), 3u);
    EXPECT_FALSE(done);
}

TEST(SimpleProcessEnvironment, ProcessValueChanges) {
    SimpleProcessEnvironment env(1.0, 1.0, 0.01, 0.0);
    env.Reset();
    double pv0 = env.GetProcessValue();
    env.Step({}, 5.0);
    double pv1 = env.GetProcessValue();
    EXPECT_NE(pv0, pv1);
}

// ======================== RlAgent ========================

TEST(RlAgent, SelectActionSize) {
    RlAgent agent(3, 3);
    auto action = agent.SelectAction({1.0, 0.0, 0.0});
    EXPECT_EQ(action.size(), 3u);
}

TEST(RlAgent, EpsilonDecays) {
    RlAgent agent(3, 3, 8, 0.001, 0.99, 1.0, 0.9, 0.01);
    double e0 = agent.GetEpsilon();
    agent.Learn({1, 0, 0}, {0.5, 0.1, 0.05}, -1.0, {0.5, 0.1, 0.3}, false);
    double e1 = agent.GetEpsilon();
    EXPECT_LT(e1, e0);
}

// ======================== GridSearch ========================

TEST(RlAgent, GridSearchReturnsThreeParams) {
    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    RlAgent agent(3, 3);
    auto params = agent.GridSearch(env, 20);
    EXPECT_EQ(params.size(), 3u);
    EXPECT_GE(params[0], 0.0);
}

// ======================== SelfTuningPidController ========================

TEST(SelfTuningPidController, RunReturnsHistory) {
    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto pid = std::make_shared<PidController>(0.5, 0.1, 0.05, 0.01);
    SelfTuningPidController stc(env, pid);
    auto history = stc.Run(50, 1.0);
    EXPECT_EQ(history.size(), 50u);
    EXPECT_EQ(history[0].size(), 4u);
}
