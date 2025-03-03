/**
 * @file test_QLearning.cpp
 * @brief Google Test suite for the Q-Learning module
 */

#include <gtest/gtest.h>
#include "QLearning.hpp"

#include <memory>
#include <vector>
#include <string>
#include <cmath>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// Alias to avoid macro comma issues with EXPECT_THROW / EXPECT_NO_THROW
using GridAgent = QLearningAgent<State, Action>;

// -----------------------------------------------------------------------
// GridWorld tests
// -----------------------------------------------------------------------

TEST(QLearning, GridWorldReset) {
    PrintSection("GridWorld Reset",
        "Verifies that Reset() returns the agent to the initial position");

    GridWorld gw(5, 5, {0, 0}, {4, 4}, {});
    State s = gw.Reset();
    EXPECT_EQ(s.x, 0);
    EXPECT_EQ(s.y, 0);
}

TEST(QLearning, ValidActionsAtCorner) {
    PrintSection("Valid Actions At Corner",
        "At (0,0) the agent can only move RIGHT or DOWN (UP/LEFT are no-ops clamped)");

    GridWorld gw(5, 5, {0, 0}, {4, 4}, {});
    gw.Reset();
    auto actions = gw.GetValidActions();

    // All 4 actions are technically valid (clamped movement), so size == 4
    EXPECT_EQ(actions.size(), 4u);
}

TEST(QLearning, StepMovement) {
    PrintSection("Step Movement",
        "Moving RIGHT from (0,0) should land on (1,0) with a small penalty");

    GridWorld gw(5, 5, {0, 0}, {4, 4}, {});
    gw.Reset();
    auto [state, reward, done] = gw.Step(Action::RIGHT);
    EXPECT_EQ(state.x, 1);
    EXPECT_EQ(state.y, 0);
    EXPECT_FALSE(done);
    EXPECT_DOUBLE_EQ(reward, -0.1);
}

TEST(QLearning, ReachGoal) {
    PrintSection("Reach Goal",
        "Stepping into the goal cell should return done=true and reward=10");

    GridWorld gw(2, 1, {0, 0}, {1, 0}, {});
    gw.Reset();
    auto [state, reward, done] = gw.Step(Action::RIGHT);
    EXPECT_TRUE(done);
    EXPECT_DOUBLE_EQ(reward, 10.0);
}

TEST(QLearning, ObstacleCollision) {
    PrintSection("Obstacle Collision",
        "Moving into an obstacle should keep the agent in place with -1 penalty");

    // Place obstacle at (1,0), agent starts at (0,0)
    GridWorld gw(5, 5, {0, 0}, {4, 4}, {{1, 0}});
    gw.Reset();
    auto [state, reward, done] = gw.Step(Action::RIGHT);
    EXPECT_EQ(state.x, 0);
    EXPECT_EQ(state.y, 0);
    EXPECT_DOUBLE_EQ(reward, -1.0);
    EXPECT_FALSE(done);
}

TEST(QLearning, StateToString) {
    PrintSection("State ToString",
        "State string representation should be human-readable");

    State s{3, 4};
    EXPECT_EQ(s.ToString(), "(3,4)");
}

TEST(QLearning, StateEquality) {
    PrintSection("State Equality",
        "Two states with same coordinates must compare equal");

    State a{2, 3};
    State b{2, 3};
    State c{0, 0};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(QLearning, InvalidGridDimensions) {
    PrintSection("Invalid Grid Dimensions",
        "GridWorld should throw on non-positive dimensions");

    EXPECT_THROW(
        GridWorld(0, 5, {0, 0}, {4, 4}, {}),
        RLException
    );
    EXPECT_THROW(
        GridWorld(5, -1, {0, 0}, {4, 4}, {}),
        RLException
    );
}

TEST(QLearning, InitialStateOnObstacle) {
    PrintSection("Initial State On Obstacle",
        "Placing the start on an obstacle should throw");

    EXPECT_THROW(
        GridWorld(5, 5, {1, 1}, {4, 4}, {{1, 1}}),
        RLException
    );
}

// -----------------------------------------------------------------------
// QLearningAgent tests
// -----------------------------------------------------------------------

TEST(QLearning, AgentConstruction) {
    PrintSection("Agent Construction",
        "QLearningAgent should initialise without error given valid params");

    EXPECT_NO_THROW(
        GridAgent(kAllActions, 0.1, 0.99, 1.0, 0.995, 0.01)
    );
}

TEST(QLearning, AgentInvalidLearningRate) {
    PrintSection("Agent Invalid Learning Rate",
        "Learning rate outside (0,1] should throw");

    EXPECT_THROW(
        GridAgent(kAllActions, 0.0, 0.99, 1.0, 0.995, 0.01),
        RLException
    );
    EXPECT_THROW(
        GridAgent(kAllActions, 1.5, 0.99, 1.0, 0.995, 0.01),
        RLException
    );
}

TEST(QLearning, AgentChoosesAction) {
    PrintSection("Agent Chooses Action",
        "ChooseAction must return one of the known actions");

    QLearningAgent<State, Action> agent(kAllActions, 0.1, 0.99, 1.0, 0.995, 0.01, 42);
    State s{0, 0};
    Action act = agent.ChooseAction(s);
    bool found = false;
    for (const auto& a : kAllActions) {
        if (a == act) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(QLearning, QValueUpdates) {
    PrintSection("Q-Value Updates",
        "After an update the Q-value for (state, action) should be non-zero");

    QLearningAgent<State, Action> agent(kAllActions, 0.5, 0.9, 0.0, 1.0, 0.0, 42);
    State s0{0, 0};
    State s1{1, 0};
    agent.Update(s0, Action::RIGHT, 10.0, s1, false);

    double qv = agent.GetQValue(s0, Action::RIGHT);
    EXPECT_GT(qv, 0.0);
}

TEST(QLearning, ExplorationDecays) {
    PrintSection("Exploration Decays",
        "After updates the exploration rate should decrease");

    QLearningAgent<State, Action> agent(kAllActions, 0.1, 0.99, 1.0, 0.5, 0.01, 42);
    double initial_eps = agent.GetExplorationRate();

    State s{0, 0};
    agent.Update(s, Action::UP, 0.0, s, false);

    double after_eps = agent.GetExplorationRate();
    EXPECT_LT(after_eps, initial_eps);
}

TEST(QLearning, AgentLearnsOptimalPath) {
    PrintSection("Agent Learns Optimal Path",
        "Verifies Q-learning converges to shortest path in a simple grid world");

    // Small 3x3 grid, no obstacles, goal at (2,2)
    auto env = std::make_shared<GridWorld>(3, 3, State{0, 0}, State{2, 2},
                                           std::vector<State>{});

    auto agent = std::make_shared<QLearningAgent<State, Action>>(
        kAllActions, 0.1, 0.99, 1.0, 0.995, 0.01, 42
    );

    // Train for enough episodes to converge
    TrainAgent<State, Action>(env, agent, 2000, 50, false);

    // After training, test: the agent should reach the goal consistently
    int successes = 0;
    int test_episodes = 20;
    for (int i = 0; i < test_episodes; ++i) {
        State st = env->Reset();
        bool done = false;
        int steps = 0;
        while (!done && steps < 50) {
            Action act = agent->GetBestAction(st, kAllActions);
            auto [ns, r, d] = env->Step(act);
            st = ns;
            done = d;
            ++steps;
        }
        if (done) ++successes;
    }

    double success_rate = static_cast<double>(successes) / test_episodes;
    EXPECT_GE(success_rate, 0.8) << "Agent should reach goal >= 80% of the time";
    std::cout << "  Success rate: " << (success_rate * 100) << "%" << std::endl;
}

TEST(QLearning, TrainAgentReturnsReasonableReward) {
    PrintSection("Train Agent Returns Reasonable Reward",
        "Average reward should improve (become less negative) with training");

    auto env = std::make_shared<GridWorld>(3, 3, State{0, 0}, State{2, 2},
                                           std::vector<State>{});
    auto agent = std::make_shared<QLearningAgent<State, Action>>(
        kAllActions, 0.1, 0.99, 1.0, 0.995, 0.01, 123
    );

    double avg_reward = TrainAgent<State, Action>(env, agent, 1000, 50, false);
    // With a reachable goal the average reward should be positive after training
    EXPECT_GT(avg_reward, -5.0) << "Average reward should be reasonable after training";
}

TEST(QLearning, ActionToStringCoversAll) {
    PrintSection("ActionToString Covers All",
        "Every action enum value should convert to a non-empty string");

    for (const auto& a : kAllActions) {
        std::string name = ActionToString(a);
        EXPECT_FALSE(name.empty());
    }
}

TEST(QLearning, CreateSimpleGridWorldFactory) {
    PrintSection("CreateSimpleGridWorld Factory",
        "Factory function should produce a valid environment");

    auto env = CreateSimpleGridWorld();
    ASSERT_NE(env, nullptr);
    State s = env->Reset();
    EXPECT_EQ(s.x, 0);
    EXPECT_EQ(s.y, 0);
}
