/**
 * @file PerceptronAlgorithm.cpp
 * @brief Implementation of the perceptron learning algorithm.
 */

#include "PerceptronAlgorithm.hpp"

PerceptronAlgorithm::PerceptronAlgorithm(double learningRate, int epochs,
                                           double w1Init, double w2Init,
                                           double biasInit)
    : w1_(w1Init), w2_(w2Init), bias_(biasInit),
      learningRate_(learningRate), epochs_(epochs) {}

int PerceptronAlgorithm::StepFunction(double sum) const {
    return (sum >= 0) ? 1 : 0;
}

void PerceptronAlgorithm::Train(const std::vector<DataPoint>& dataset) {
    for (int epoch = 0; epoch < epochs_; ++epoch) {
        for (const auto& dp : dataset) {
            double sum = w1_ * dp.X1 + w2_ * dp.X2 + bias_;
            int prediction = StepFunction(sum);
            int error = dp.Label - prediction;

            if (error != 0) {
                w1_ += learningRate_ * error * dp.X1;
                w2_ += learningRate_ * error * dp.X2;
                bias_ += learningRate_ * error;
            }
        }
    }
}

int PerceptronAlgorithm::Predict(double x1, double x2) const {
    double sum = w1_ * x1 + w2_ * x2 + bias_;
    return StepFunction(sum);
}

double PerceptronAlgorithm::GetW1() const { return w1_; }
double PerceptronAlgorithm::GetW2() const { return w2_; }
double PerceptronAlgorithm::GetBias() const { return bias_; }
