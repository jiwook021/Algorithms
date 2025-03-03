/**
 * @file QLearning.cpp
 * @brief Implementation of the Q-Learning demo with visual output
 */

#include "QLearning.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void PrintHeader(const std::string& title) {
    std::string rule(60, '=');
    std::cout << "\n" << rule << "\n"
              << "  " << title << "\n"
              << rule << "\n";
}

void PrintPolicy(
    const std::shared_ptr<GridWorld>& env,
    const std::shared_ptr<QLearningAgent<State, Action>>& agent
) {
    int w = env->GetWidth();
    int h = env->GetHeight();
    State goal = env->GetGoalState();

    std::cout << "    ";
    for (int c = 0; c < w; ++c) std::cout << " " << c << "  ";
    std::cout << "\n";

    std::string sep = "  +";
    for (int c = 0; c < w; ++c) sep += "---+";

    for (int r = 0; r < h; ++r) {
        std::cout << sep << "\n";
        std::cout << r << " |";
        for (int c = 0; c < w; ++c) {
            State s{c, r};
            if (s == goal) {
                std::cout << " G |";
            } else if (env->IsObstacle(s)) {
                std::cout << " # |";
            } else {
                Action best = agent->GetBestAction(s, kAllActions);
                std::cout << " " << ActionToArrow(best) << " |";
            }
        }
        std::cout << "\n";
    }
    std::cout << sep << "\n";
}

} // namespace

void RunReinforcementLearningDemo() {
    try {
        PrintHeader("Q-LEARNING: Grid World Navigation");

        std::cout << "\n"
                  << "  The agent (A) learns to navigate a grid to reach the goal (G)\n"
                  << "  while avoiding obstacles (#), using trial and error.\n"
                  << "\n"
                  << "  Q-Learning update rule:\n"
                  << "    Q(s,a) += lr * [reward + gamma * max Q(s',a') - Q(s,a)]\n"
                  << "\n"
                  << "  Exploration: epsilon-greedy\n"
                  << "    Starts mostly random, gradually becomes greedy as epsilon decays.\n";

        auto env = CreateSimpleGridWorld();

        auto agent = std::make_shared<QLearningAgent<State, Action>>(
            kAllActions,
            0.1,    // learning rate
            0.99,   // discount factor
            1.0,    // initial exploration rate
            0.995,  // exploration decay
            0.01    // minimum exploration rate
        );

        int training_episodes = 1000;
        int max_steps = 100;

        // --- Show initial grid ---
        PrintHeader("Grid Layout (5x5)");
        std::cout << "\n"
                  << "  A = Agent start (0,0)\n"
                  << "  G = Goal (4,4)\n"
                  << "  # = Obstacle (impassable)\n"
                  << "  . = Empty cell\n\n";
        env->Render();

        // --- Training ---
        PrintHeader("Training Phase");
        std::cout << "\n"
                  << "  Episodes:      " << training_episodes << "\n"
                  << "  Max steps:     " << max_steps << "\n"
                  << "  Learning rate: 0.10  (how fast the agent learns)\n"
                  << "  Discount:      0.99  (importance of future rewards)\n"
                  << "  Exploration:   1.00 -> 0.01  (random -> greedy)\n\n";

        auto start = std::chrono::high_resolution_clock::now();
        double avg_reward = TrainAgent<State, Action>(
            env, agent, training_episodes, max_steps, true
        );
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "\n  Done in " << std::fixed << std::setprecision(2)
                  << (ms / 1000.0) << "s | Average reward: "
                  << std::setprecision(2) << avg_reward << "\n";

        // --- Learned policy ---
        PrintHeader("Learned Policy");
        std::cout << "\n  Arrows show the best action the trained agent would take.\n\n";
        PrintPolicy(env, agent);

        // --- Testing ---
        PrintHeader("Testing Trained Agent");

        int test_episodes = 5;
        int successes = 0;
        double total_reward = 0.0;
        int total_steps = 0;

        for (int ep = 0; ep < test_episodes; ++ep) {
            std::cout << "\n  Episode " << (ep + 1) << ":\n";
            State st = env->Reset();
            double ep_reward = 0.0;
            int steps = 0;
            bool done = false;

            while (!done && steps < max_steps) {
                Action act = agent->GetBestAction(st, kAllActions);
                auto [ns, reward, is_done] = env->Step(act);

                std::cout << "    " << st.ToString()
                          << " --" << std::setw(5) << ActionToString(act) << "--> "
                          << ns.ToString()
                          << "  reward: " << std::setw(6) << std::fixed
                          << std::setprecision(2) << reward << "\n";

                st = ns;
                ep_reward += reward;
                done = is_done;
                ++steps;
            }

            if (done) {
                std::cout << "    >> GOAL REACHED in " << steps
                          << " steps | Total reward: " << std::fixed
                          << std::setprecision(2) << ep_reward << "\n";
                ++successes;
            } else {
                std::cout << "    >> FAILED after " << steps
                          << " steps | Total reward: " << std::fixed
                          << std::setprecision(2) << ep_reward << "\n";
            }

            total_reward += ep_reward;
            total_steps += steps;
        }

        // --- Summary ---
        PrintHeader("Results");
        std::cout << "\n"
                  << "  Success:        " << successes << "/" << test_episodes
                  << " episodes reached the goal ("
                  << std::fixed << std::setprecision(0)
                  << (100.0 * successes / test_episodes) << "%)\n"
                  << "  Average steps:  " << std::setprecision(1)
                  << (static_cast<double>(total_steps) / test_episodes) << "\n"
                  << "  Average reward: " << std::setprecision(2)
                  << (total_reward / test_episodes) << "\n\n";

    } catch (const RLException& e) {
        std::cerr << "RL Error: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << '\n';
    }
}
