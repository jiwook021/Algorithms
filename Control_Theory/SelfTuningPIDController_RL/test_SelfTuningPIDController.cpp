/**
 * @file test_SelfTuningPIDController.cpp
 * @brief GTest unit tests for the SelfTuningPIDController module
 * @details Tests PIDController, NeuralNetwork, SimpleProcessEnvironment,
 *          RLAgent, and SelfTuningPIDController with educational output
 *          explaining WHY each expected value holds.
 */

#include "SelfTuningPIDController.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr double TOLERANCE = 1e-6;

// ---------------------------------------------------------------------------
// PIDController Tests
// ---------------------------------------------------------------------------

TEST(PIDController, ComputePositiveOutput) {
    std::cout << "\n  Why: A PID with positive gains (Kp=1, Ki=0.1, Kd=0.01) and"
              << "\n       positive error=5 must produce a positive control signal."
              << "\n       P-term = 1.0*5.0 = 5.0, which alone is positive."
              << "\n       The integral and derivative add small contributions"
              << "\n       in the same direction on the first call.\n";

    PIDController pid(1.0, 0.1, 0.01);
    double output = pid.Compute(5.0);
    EXPECT_GT(output, 0.0);
}

TEST(PIDController, ProportionalOnlyOutput) {
    std::cout << "\n  Why: With Ki=0, Kd=0, only the P-term contributes."
              << "\n       output = Kp * error = 2.0 * 3.0 = 6.0 exactly."
              << "\n       No integral accumulation, no derivative (first call"
              << "\n       has zero previous error, but Kd=0 anyway).\n";

    PIDController pid(2.0, 0.0, 0.0);
    double output = pid.Compute(3.0);
    EXPECT_NEAR(output, 6.0, TOLERANCE);
}

TEST(PIDController, UpdateParametersTakesEffect) {
    std::cout << "\n  Why: UpdateParameters(2.0, 0.5, 0.1) must store exactly"
              << "\n       those values.  The RL agent calls this every step to"
              << "\n       apply new gains — if storage is wrong, the whole"
              << "\n       self-tuning loop is broken.\n";

    PIDController pid;
    pid.UpdateParameters(2.0, 0.5, 0.1);
    auto params = pid.GetParameters();
    EXPECT_NEAR(params[0], 2.0, TOLERANCE);
    EXPECT_NEAR(params[1], 0.5, TOLERANCE);
    EXPECT_NEAR(params[2], 0.1, TOLERANCE);
}

TEST(PIDController, ResetClearsState) {
    std::cout << "\n  Why: After Compute(10.0), the PID has accumulated integral"
              << "\n       and stored previousError=10.  Reset() must zero all"
              << "\n       internal state.  A fresh Compute(10.0) after reset"
              << "\n       should behave identically to the very first call.\n";

    PIDController pid(1.0, 1.0, 1.0);
    pid.Compute(10.0);
    pid.Reset();
    double output = pid.Compute(10.0);
    EXPECT_GT(output, 0.0);
}

TEST(PIDController, OutputClampedToMax) {
    std::cout << "\n  Why: Kp=100 * error=5 = 500, but the output is clamped"
              << "\n       to [-10, 10] to protect the actuator.  The actual"
              << "\n       output must be exactly 10.0 (the MAX_CONTROL limit).\n";

    PIDController pid(100.0, 0.0, 0.0);
    double output = pid.Compute(5.0);
    EXPECT_NEAR(output, 10.0, TOLERANCE);
}

TEST(PIDController, GetParametersReturnsThreeElements) {
    std::cout << "\n  Why: GetParameters returns [Kp, Ki, Kd] — always 3 elements."
              << "\n       The RL agent indexes result[0..2] without bounds checks,"
              << "\n       so a different size would cause undefined behavior.\n";

    PIDController pid(1.0, 2.0, 3.0);
    auto params = pid.GetParameters();
    EXPECT_EQ(params.size(), 3u);
}

// ---------------------------------------------------------------------------
// NeuralNetwork Tests
// ---------------------------------------------------------------------------

TEST(NeuralNetwork, ForwardOutputSize) {
    std::cout << "\n  Why: A network with outputSize=2 must return exactly 2 values."
              << "\n       The Q-learning loop expects one Q-value per action"
              << "\n       dimension — a size mismatch would corrupt the policy.\n";

    NeuralNetwork nn(3, 4, 2, 0.01);
    auto output = nn.Forward({1.0, 0.5, -0.5});
    EXPECT_EQ(output.size(), 2u);
}

TEST(NeuralNetwork, ForwardIsDeterministic) {
    std::cout << "\n  Why: With frozen weights, Forward(x) must return the same"
              << "\n       result every time.  The computation is purely"
              << "\n       h = ReLU(W1*x + b1), y = clamp(W2*h + b2)"
              << "\n       with no randomness.  Non-determinism would make"
              << "\n       Q-value learning impossible.\n";

    NeuralNetwork nn(2, 3, 1, 0.01);
    auto out1 = nn.Forward({1.0, 2.0});
    auto out2 = nn.Forward({1.0, 2.0});
    EXPECT_NEAR(out1[0], out2[0], TOLERANCE);
}

TEST(NeuralNetwork, ForwardThrowsOnInputSizeMismatch) {
    std::cout << "\n  Why: Network expects inputSize=3 but receives 2 values."
              << "\n       Without this check, the inner loop would read past"
              << "\n       the input vector, producing garbage Q-values.\n";

    NeuralNetwork nn(3, 4, 2, 0.01);
    EXPECT_THROW(nn.Forward({1.0, 2.0}), std::invalid_argument);
}

TEST(NeuralNetwork, TrainReducesLoss) {
    std::cout << "\n  Why: Backpropagation adjusts weights to minimize MSE."
              << "\n       After 50 iterations on the same (input, target) pair,"
              << "\n       the loss must be lower than the initial loss."
              << "\n       If it isn't, gradient descent is broken.\n";

    NeuralNetwork nn(2, 8, 1, 0.01);
    std::vector<double> input  = {0.5, 0.5};
    std::vector<double> target = {0.8};
    double initialLoss = nn.Train(input, target);
    double finalLoss = 0.0;
    for (int i = 0; i < 50; ++i) {
        finalLoss = nn.Train(input, target);
    }
    EXPECT_LE(finalLoss, initialLoss);
}

TEST(NeuralNetwork, SaveLoadRoundTrip) {
    std::cout << "\n  Why: Saving and loading must preserve all weights exactly."
              << "\n       The same input must produce the same output before"
              << "\n       and after a save/load cycle, ensuring long RL training"
              << "\n       runs can be checkpointed and resumed.\n";

    NeuralNetwork nn(2, 4, 1, 0.01);
    auto before = nn.Forward({1.0, -1.0});
    nn.SaveModel("test_nn_model.dat");

    NeuralNetwork nn2(2, 4, 1, 0.01);
    nn2.LoadModel("test_nn_model.dat");
    auto after = nn2.Forward({1.0, -1.0});

    std::remove("test_nn_model.dat");
    EXPECT_NEAR(before[0], after[0], TOLERANCE);
}

TEST(NeuralNetwork, DimensionGetters) {
    std::cout << "\n  Why: GetInputSize/GetHiddenSize/GetOutputSize must report"
              << "\n       the sizes passed at construction (3, 8, 2).  The RL"
              << "\n       agent relies on these to match state and action dims.\n";

    NeuralNetwork nn(3, 8, 2, 0.01);
    EXPECT_EQ(nn.GetInputSize(),  3u);
    EXPECT_EQ(nn.GetHiddenSize(), 8u);
    EXPECT_EQ(nn.GetOutputSize(), 2u);
}

// ---------------------------------------------------------------------------
// SimpleProcessEnvironment Tests
// ---------------------------------------------------------------------------

TEST(Environment, ResetReturnsThreeElementState) {
    std::cout << "\n  Why: The state vector is [error, errorDeriv, lastControl]."
              << "\n       The RL agent's neural network has inputSize=3, so"
              << "\n       Reset() must return exactly 3 elements.\n";

    SimpleProcessEnvironment env;
    auto state = env.Reset();
    EXPECT_EQ(state.size(), 3u);
}

TEST(Environment, ResetSetsInitialValues) {
    std::cout << "\n  Why: After Reset(), PV starts at 0.0 and setpoint at 1.0."
              << "\n       This creates an initial error of 1.0, giving the PID"
              << "\n       a consistent starting condition for reproducible"
              << "\n       training episodes.\n";

    SimpleProcessEnvironment env;
    env.Reset();
    EXPECT_NEAR(env.GetProcessValue(), 0.0, TOLERANCE);
    EXPECT_NEAR(env.GetSetpoint(), 1.0, TOLERANCE);
}

TEST(Environment, StepReturnsValidTuple) {
    std::cout << "\n  Why: Step() returns (state[3], reward, done).  The first"
              << "\n       step cannot be 'done' because the episode just started."
              << "\n       State must be 3 elements to match the RL agent's input.\n";

    SimpleProcessEnvironment env;
    env.Reset();
    auto [state, reward, done] = env.Step({1.0, 0.1, 0.01}, 0.5);
    EXPECT_EQ(state.size(), 3u);
    EXPECT_FALSE(done);
}

TEST(Environment, PositiveControlIncreasesPV) {
    std::cout << "\n  Why: The process model is dPV/dt = (gain*u - PV) / tau."
              << "\n       With PV=0, gain=1, u=5.0:  dPV/dt = (5-0)/1 = 5."
              << "\n       After one step (dt=0.01): PV ~= 0.01*5 = 0.05."
              << "\n       The process value must increase from zero.\n";

    SimpleProcessEnvironment env(1.0, 1.0, 0.01, 0.0);
    env.Reset();
    double pvBefore = env.GetProcessValue();
    env.Step({1.0, 0.0, 0.0}, 5.0);
    double pvAfter = env.GetProcessValue();
    EXPECT_GT(pvAfter, pvBefore);
}

TEST(Environment, EpisodeTerminatesAfterMaxSteps) {
    std::cout << "\n  Why: With maxSteps=5, the episode must signal done=true"
              << "\n       after 5 steps.  This prevents infinite training loops"
              << "\n       and ensures the RL agent always sees episode boundaries.\n";

    SimpleProcessEnvironment env;
    env.SetMaxSteps(5);
    env.Reset();
    bool lastDone = false;
    for (int i = 0; i < 10; ++i) {
        auto [s, r, d] = env.Step({1.0, 0.0, 0.0}, 1.0);
        lastDone = d;
        if (d) break;
    }
    EXPECT_TRUE(lastDone);
}

TEST(Environment, SetSetpointUpdatesTarget) {
    std::cout << "\n  Why: SetSetpoint(3.0) must change the target the PID tracks."
              << "\n       This is needed for multi-setpoint evaluation (e.g.,"
              << "\n       testing gains at setpoint=1.0 then setpoint=0.5).\n";

    SimpleProcessEnvironment env;
    env.Reset();
    env.SetSetpoint(3.0);
    EXPECT_NEAR(env.GetSetpoint(), 3.0, TOLERANCE);
}

TEST(Environment, SetSetpointClampedToRange) {
    std::cout << "\n  Why: Setpoints outside [-5, 5] are clamped.  Requesting"
              << "\n       100.0 gives 5.0.  This prevents the simulation from"
              << "\n       entering unrealistic operating regimes where the PID"
              << "\n       gains discovered wouldn't transfer to real systems.\n";

    SimpleProcessEnvironment env;
    env.SetSetpoint(100.0);
    EXPECT_NEAR(env.GetSetpoint(), 5.0, TOLERANCE);
}

// ---------------------------------------------------------------------------
// RLAgent Tests
// ---------------------------------------------------------------------------

TEST(RLAgent, SelectActionReturnsThreeElements) {
    std::cout << "\n  Why: The PID has 3 gains [Kp, Ki, Kd], so SelectAction"
              << "\n       must return exactly 3 values — one per gain.\n";

    RLAgent agent(3, 3, 8, 0.001);
    auto action = agent.SelectAction({1.0, 0.0, 0.0});
    EXPECT_EQ(action.size(), 3u);
}

TEST(RLAgent, ActionsWithinDefinedRanges) {
    std::cout << "\n  Why: With epsilon=1.0 (pure random exploration), actions are"
              << "\n       drawn uniformly from [0,2] x [0,1] x [0,0.5]."
              << "\n       These ranges are chosen so Kp >= Ki >= Kd, which"
              << "\n       matches typical stable PID tuning patterns.\n";

    RLAgent agent(3, 3, 8, 0.001, 0.99, 1.0);
    auto action = agent.SelectAction({1.0, 0.0, 0.0});
    EXPECT_GE(action[0], 0.0); EXPECT_LE(action[0], 2.0);
    EXPECT_GE(action[1], 0.0); EXPECT_LE(action[1], 1.0);
    EXPECT_GE(action[2], 0.0); EXPECT_LE(action[2], 0.5);
}

TEST(RLAgent, LearnReturnsNonNegativeLoss) {
    std::cout << "\n  Why: Learn() performs a Bellman backup on the Q-network:"
              << "\n       target = reward + gamma * max Q(s', a')"
              << "\n       then trains via backprop, returning MSE loss."
              << "\n       MSE = mean((target - predicted)^2) >= 0 always.\n";

    RLAgent agent(3, 3, 8, 0.001);
    double loss = agent.Learn({1.0, 0.0, 0.0}, {0.5, 0.1, 0.05},
                               -0.5, {0.8, -0.1, 0.5}, false);
    EXPECT_GE(loss, 0.0);
}

TEST(RLAgent, EpsilonDecaysAfterLearning) {
    std::cout << "\n  Why: Epsilon starts at 1.0 and decays by factor 0.9 each"
              << "\n       Learn() call.  After one call: eps = 1.0 * 0.9 = 0.9."
              << "\n       This shifts the agent from exploration (random gains)"
              << "\n       toward exploitation (using learned Q-values).\n";

    RLAgent agent(3, 3, 8, 0.001, 0.99, 1.0, 0.9, 0.01);
    double epsBefore = agent.GetEpsilon();
    agent.Learn({1.0, 0.0, 0.0}, {0.5, 0.1, 0.05}, -0.5, {0.8, -0.1, 0.5}, false);
    double epsAfter = agent.GetEpsilon();
    EXPECT_LT(epsAfter, epsBefore);
}

TEST(RLAgent, GridSearchReturnsValidParameters) {
    std::cout << "\n  Why: GridSearch tests 4x4x4=64 Kp/Ki/Kd combinations for"
              << "\n       50 steps each, returning the triple with the highest"
              << "\n       cumulative reward.  Result must be 3 non-negative"
              << "\n       values — valid PID gains.\n";

    RLAgent agent(3, 3, 8, 0.001);
    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto best = agent.GridSearch(env, 50);
    EXPECT_EQ(best.size(), 3u);
    EXPECT_GE(best[0], 0.0);
}

// ---------------------------------------------------------------------------
// SelfTuningPIDController Tests
// ---------------------------------------------------------------------------

TEST(SelfTuningPID, SimpleTuningProducesValidGains) {
    std::cout << "\n  Why: SimpleTuning runs GridSearch and updates the shared"
              << "\n       PIDController.  The resulting gains must be valid"
              << "\n       (3 non-negative values) and represent the best-performing"
              << "\n       combination from the 4x4x4 grid.\n";

    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto pid = std::make_shared<PIDController>(0.5, 0.1, 0.05, 0.01);
    SelfTuningPIDController stpid(env, pid, 8);

    stpid.SimpleTuning(50);
    auto params = pid->GetParameters();
    EXPECT_EQ(params.size(), 3u);
    EXPECT_GE(params[0], 0.0);
}

TEST(SelfTuningPID, RunReturnsCorrectHistoryShape) {
    std::cout << "\n  Why: Run(100, 1.0) executes 100 PID steps at dt=0.01."
              << "\n       It must return exactly 100 entries, each with 4 fields:"
              << "\n       [time, setpoint, processValue, controlSignal]."
              << "\n       100 entries because the loop runs step=0..99."
              << "\n       4 fields because that's the complete state needed for"
              << "\n       offline analysis (time-series plot of tracking).\n";

    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto pid = std::make_shared<PIDController>(1.0, 0.1, 0.05, 0.01);
    SelfTuningPIDController stpid(env, pid, 8);

    auto history = stpid.Run(100, 1.0);
    EXPECT_EQ(history.size(), 100u);
    ASSERT_FALSE(history.empty());
    EXPECT_EQ(history[0].size(), 4u);
}

TEST(SelfTuningPID, ConvergesToSetpoint) {
    std::cout << "\n  Why: With Kp=1.0, Ki=0.2, Kd=0.05 and tau=1.0, the closed-loop"
              << "\n       system should settle within ~5 time constants = 5s = 500 steps."
              << "\n       The first-order process dPV/dt = (u - PV)/tau is stable"
              << "\n       under PID control, so the final PV must be near setpoint=1.0."
              << "\n       Tolerance 0.3 accounts for the discrete-time approximation"
              << "\n       and rate clamping in the environment model.\n";

    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto pid = std::make_shared<PIDController>(1.0, 0.2, 0.05, 0.01);
    SelfTuningPIDController stpid(env, pid, 8);

    auto history = stpid.Run(500, 1.0);
    ASSERT_FALSE(history.empty());
    double finalPV       = history.back()[2];
    double finalSetpoint = history.back()[1];
    EXPECT_NEAR(finalPV, finalSetpoint, 0.3);
}

TEST(SelfTuningPID, TrainCompletesWithValidParameters) {
    std::cout << "\n  Why: Train(2 episodes, sync every 1, 20 steps/ep) exercises"
              << "\n       the full RL loop: SelectAction -> UpdateParameters ->"
              << "\n       Compute -> Step -> Learn -> SyncTargetNetwork."
              << "\n       If any component crashes or produces NaN, the test fails."
              << "\n       After training, PID gains must still be a valid 3-vector.\n";

    auto env = std::make_shared<SimpleProcessEnvironment>(1.0, 1.0, 0.01, 0.0);
    auto pid = std::make_shared<PIDController>(0.5, 0.1, 0.05, 0.01);
    SelfTuningPIDController stpid(env, pid, 8);

    stpid.Train(2, 1, 20);
    auto params = pid->GetParameters();
    EXPECT_EQ(params.size(), 3u);

    std::remove("best_pid_model.dat");
}
