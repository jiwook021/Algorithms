/**
 * @file main.cpp
 * @brief Demo and benchmark driver for the DecisionTree implementation.
 */
#include "DecisionTree.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

// Helper: print a confusion matrix
template<typename T>
void PrintConfusionMatrix(const std::vector<T>& TrueLabels, const std::vector<T>& Predictions) {
    std::vector<T> UniqueLabels = TrueLabels;
    std::sort(UniqueLabels.begin(), UniqueLabels.end());
    auto Last = std::unique(UniqueLabels.begin(), UniqueLabels.end());
    UniqueLabels.erase(Last, UniqueLabels.end());

    std::cout << "\nConfusion Matrix:\n";
    std::cout << std::setw(10) << "Actual\\Pred";
    for (const auto& Label : UniqueLabels) {
        std::cout << std::setw(10) << Label;
    }
    std::cout << "\n";

    for (const auto& TrueLabel : UniqueLabels) {
        std::cout << std::setw(10) << TrueLabel;
        for (const auto& PredLabel : UniqueLabels) {
            size_t count = 0;
            for (size_t i = 0; i < TrueLabels.size(); ++i) {
                if (TrueLabels[i] == TrueLabel && Predictions[i] == PredLabel) {
                    count++;
                }
            }
            std::cout << std::setw(10) << count;
        }
        std::cout << "\n";
    }
}

// Helper: calculate classification accuracy
template<typename T>
double CalculateAccuracy(const std::vector<T>& TrueLabels, const std::vector<T>& Predictions) {
    if (TrueLabels.size() != Predictions.size() || TrueLabels.empty()) {
        return 0.0;
    }
    size_t Correct = 0;
    for (size_t i = 0; i < TrueLabels.size(); ++i) {
        if (TrueLabels[i] == Predictions[i]) {
            Correct++;
        }
    }
    return static_cast<double>(Correct) / TrueLabels.size();
}

// Generate a simple 2-feature dataset
Ml::DecisionTree<double>::Dataset GenerateSimpleDataset() {
    const size_t NSamples = 1000;
    std::vector<std::vector<double>> Features(NSamples, std::vector<double>(2));
    std::vector<double> Labels(NSamples);
    std::random_device Rd;
    std::mt19937 Gen(Rd());
    std::uniform_real_distribution<> Dis(0.0, 2.0);

    for (size_t i = 0; i < NSamples; ++i) {
        Features[i][0] = Dis(Gen);
        Features[i][1] = Dis(Gen);
        Labels[i] = (Features[i][0] + Features[i][1] <= 1.0) ? 0.0 : 1.0;
    }
    return {Features, Labels, {"x", "y"}};
}

// Generate a simplified Iris-like dataset
Ml::DecisionTree<double>::Dataset GenerateIrisDataset() {
    const size_t NSamples = 150;
    std::vector<std::vector<double>> Features(NSamples, std::vector<double>(3));
    std::vector<double> Labels(NSamples);

    for (size_t i = 0; i < 50; ++i) {
        Features[i][0] = 5.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][1] = 3.5 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][2] = 1.5 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Labels[i] = 0.0;
    }
    for (size_t i = 50; i < 100; ++i) {
        Features[i][0] = 6.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][1] = 2.5 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][2] = 4.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Labels[i] = 1.0;
    }
    for (size_t i = 100; i < 150; ++i) {
        Features[i][0] = 6.5 + 0.6 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][1] = 3.0 + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Features[i][2] = 5.5 + 0.6 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        Labels[i] = 2.0;
    }
    return {Features, Labels, {"sepal_length", "sepal_width", "petal_length"}};
}

// Train/test split
template<typename T>
std::pair<typename Ml::DecisionTree<T>::Dataset, typename Ml::DecisionTree<T>::Dataset>
TrainTestSplit(const typename Ml::DecisionTree<T>::Dataset& DatasetIn, double TestSize = 0.2) {
    if (DatasetIn.Features.empty() || DatasetIn.Labels.empty()) {
        throw std::invalid_argument("Empty dataset");
    }

    size_t NSamples = DatasetIn.Features.size();
    size_t NTest = static_cast<size_t>(NSamples * TestSize);
    size_t NTrain = NSamples - NTest;

    std::vector<size_t> Indices(NSamples);
    std::iota(Indices.begin(), Indices.end(), 0);
    std::random_device Rd;
    std::mt19937 g(Rd());
    std::shuffle(Indices.begin(), Indices.end(), g);

    typename Ml::DecisionTree<T>::Dataset TrainDs, TestDs;
    TrainDs.Features.resize(NTrain);
    TrainDs.Labels.resize(NTrain);
    TrainDs.FeatureNames = DatasetIn.FeatureNames;
    TestDs.Features.resize(NTest);
    TestDs.Labels.resize(NTest);
    TestDs.FeatureNames = DatasetIn.FeatureNames;

    for (size_t i = 0; i < NTrain; ++i) {
        TrainDs.Features[i] = DatasetIn.Features[Indices[i]];
        TrainDs.Labels[i] = DatasetIn.Labels[Indices[i]];
    }
    for (size_t i = 0; i < NTest; ++i) {
        TestDs.Features[i] = DatasetIn.Features[Indices[NTrain + i]];
        TestDs.Labels[i] = DatasetIn.Labels[Indices[NTrain + i]];
    }
    return {TrainDs, TestDs};
}

// Run a demo on a given dataset
void RunDemo(const std::string& Name,
              const Ml::DecisionTree<double>::Dataset& DatasetIn,
              Ml::DecisionTree<double>::Config Config) {
    std::cout << "\n===== " << Name << " =====\n";
    auto [TrainDs, TestDs] = TrainTestSplit<double>(DatasetIn, 0.2);
    std::cout << "Train: " << TrainDs.Features.size()
              << "  Test: " << TestDs.Features.size() << "\n";

    Ml::DecisionTree<double> Tree(Config);

    auto T0 = std::chrono::high_resolution_clock::now();
    Tree.Fit(TrainDs);
    auto T1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training:   "
              << std::chrono::duration<double, std::milli>(T1 - T0).count() << " ms\n";

    T0 = std::chrono::high_resolution_clock::now();
    auto Preds = Tree.Predict(TestDs.Features);
    T1 = std::chrono::high_resolution_clock::now();
    std::cout << "Prediction: "
              << std::chrono::duration<double, std::milli>(T1 - T0).count() << " ms\n";

    double Acc = CalculateAccuracy(TestDs.Labels, Preds);
    std::cout << "Accuracy:   " << std::fixed << std::setprecision(2) << Acc * 100 << "%\n";

    PrintConfusionMatrix(TestDs.Labels, Preds);

    std::cout << "Depth: " << Tree.GetDepth()
              << "  Nodes: " << Tree.GetNodeCount() << "\n";

    auto Imp = Tree.FeatureImportance();
    std::cout << "Feature importance:\n";
    for (size_t i = 0; i < Imp.size(); ++i) {
        std::cout << "  " << DatasetIn.FeatureNames[i] << ": "
                  << std::fixed << std::setprecision(4) << Imp[i] << "\n";
    }
}

int main() {
    std::cout << "Decision Tree -- Demo & Benchmark\n";
    std::cout << "=================================\n";
    std::srand(42);

    // Demo: simple dataset (Gini)
    {
        Ml::DecisionTree<double>::Config Cfg;
        Cfg.SplitCriterion = Ml::DecisionTree<double>::Criterion::Gini;
        Cfg.MaxDepth = 5;
        RunDemo("Simple Dataset (Gini)", GenerateSimpleDataset(), Cfg);
    }

    // Demo: Iris dataset (Entropy)
    {
        Ml::DecisionTree<double>::Config Cfg;
        Cfg.SplitCriterion = Ml::DecisionTree<double>::Criterion::Entropy;
        Cfg.MaxDepth = 4;
        RunDemo("Iris Dataset (Entropy)", GenerateIrisDataset(), Cfg);
    }

    // Benchmark: large dataset with varying depths
    std::cout << "\n===== Performance Benchmark =====\n";
    const size_t NSamples = 10000;
    const size_t NFeatures = 5;
    std::vector<std::vector<double>> Feats(NSamples, std::vector<double>(NFeatures));
    std::vector<double> Labels(NSamples);
    std::random_device Rd;
    std::mt19937 Gen(Rd());
    std::uniform_real_distribution<> Dis(0.0, 10.0);

    for (size_t i = 0; i < NSamples; ++i) {
        double Sum = 0.0;
        for (size_t j = 0; j < NFeatures; ++j) {
            Feats[i][j] = Dis(Gen);
            Sum += Feats[i][j];
        }
        Labels[i] = (Sum > 25.0) ? 1.0 : 0.0;
    }

    Ml::DecisionTree<double>::Dataset LargeDs{Feats, Labels, {"f1","f2","f3","f4","f5"}};
    auto [TrainDs, TestDs] = TrainTestSplit<double>(LargeDs, 0.2);

    for (size_t Depth : {5u, 10u, 15u, 20u}) {
        Ml::DecisionTree<double>::Config Cfg;
        Cfg.MaxDepth = Depth;
        Ml::DecisionTree<double> Tree(Cfg);

        auto T0 = std::chrono::high_resolution_clock::now();
        Tree.Fit(TrainDs);
        auto T1 = std::chrono::high_resolution_clock::now();

        auto Preds = Tree.Predict(TestDs.Features);
        auto T2 = std::chrono::high_resolution_clock::now();

        double Acc = CalculateAccuracy(TestDs.Labels, Preds);
        std::cout << "\nmax_depth=" << Depth
                  << "  train=" << std::chrono::duration<double, std::milli>(T1-T0).count() << "ms"
                  << "  predict=" << std::chrono::duration<double, std::milli>(T2-T1).count() << "ms"
                  << "  acc=" << std::fixed << std::setprecision(2) << Acc*100 << "%"
                  << "  depth=" << Tree.GetDepth()
                  << "  nodes=" << Tree.GetNodeCount() << "\n";
    }

    return 0;
}
