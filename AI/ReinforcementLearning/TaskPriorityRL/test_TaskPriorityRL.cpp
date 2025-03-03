#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <random>
#include <cmath>

class NeuralNetwork {
    std::vector<std::vector<std::vector<float>>> Weights;
    std::vector<std::vector<float>> Biases;
    float LearningRate;
    static float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
public:
    NeuralNetwork(const std::vector<size_t>& layerSizes, float learningRate = 0.01f)
        : LearningRate(learningRate) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0, 0.1f);
        for (size_t i = 1; i < layerSizes.size(); i++) {
            std::vector<std::vector<float>> w(layerSizes[i], std::vector<float>(layerSizes[i-1]));
            for (auto& row : w) for (auto& v : row) v = dist(gen);
            Weights.push_back(w);
            Biases.push_back(std::vector<float>(layerSizes[i], 0));
        }
    }
    float Predict(const std::vector<float>& inputs) const {
        std::vector<float> act = inputs;
        for (size_t l = 0; l < Weights.size(); l++) {
            std::vector<float> next(Weights[l].size(), 0);
            for (size_t i = 0; i < Weights[l].size(); i++) {
                for (size_t j = 0; j < act.size(); j++) next[i] += Weights[l][i][j] * act[j];
                next[i] += Biases[l][i];
                next[i] = Sigmoid(next[i]);
            }
            act = next;
        }
        return act[0];
    }
};

struct Task {
    int Id; std::string Description; float EstimatedHours;
    int GetId() const { return Id; }
    std::string GetDescription() const { return Description; }
    float GetEstimatedHours() const { return EstimatedHours; }
    std::vector<float> ToFeatureVector() const {
        return {(float)Id, EstimatedHours, (float)Description.size()};
    }
};

TEST(TaskPriorityRL, NNPredict) {
    NeuralNetwork nn({3, 4, 1}, 0.01f);
    float out = nn.Predict({1.0f, 0.5f, 0.3f});
    EXPECT_GE(out, 0.0f); EXPECT_LE(out, 1.0f);
}

TEST(TaskPriorityRL, NNDeterministic) {
    NeuralNetwork nn({2, 3, 1}, 0.01f);
    float o1 = nn.Predict({1.0f, 2.0f});
    float o2 = nn.Predict({1.0f, 2.0f});
    EXPECT_FLOAT_EQ(o1, o2);
}

TEST(TaskPriorityRL, TaskFeatureVector) {
    Task t{1, "Fix bug", 2.5f};
    auto fv = t.ToFeatureVector();
    EXPECT_EQ(fv.size(), 3u);
    EXPECT_FLOAT_EQ(fv[0], 1.0f);
    EXPECT_FLOAT_EQ(fv[1], 2.5f);
}

TEST(TaskPriorityRL, TaskAccessors) {
    Task t{42, "Deploy", 1.0f};
    EXPECT_EQ(t.GetId(), 42);
    EXPECT_EQ(t.GetDescription(), "Deploy");
}
