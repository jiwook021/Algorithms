/**
 * @file DecisionTree.hpp
 * @brief Decision Tree implementation supporting classification and regression.
 *
 * @details A fully templated Decision Tree that supports Gini impurity, Entropy,
 *          and MSE splitting criteria via the SplitCriterion concept.
 *          Thread-safe for concurrent predict/fit calls.
 *
 *          Header organisation:
 *            DecisionTreeConfig.hpp  - SplitCriterion concept, concrete criteria,
 *                                     SplitCriterionType enum, Config struct.
 *            DecisionTreeNode.hpp    - Dataset, Node struct, traversal helpers.
 *            DecisionTree.hpp        - Main DecisionTree class (this file).
 */
#pragma once

#include "DecisionTreeConfig.hpp"
#include "DecisionTreeNode.hpp"

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <optional>
#include <random>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <utility>
#include <functional>
#include <sstream>

namespace Ml {

/**
 * @brief Decision Tree supporting both categorical and numerical features.
 * @tparam T The numeric type for features and labels (default: double).
 *
 * @details Supports Gini impurity, Entropy (classification), and MSE (regression)
 *          splitting criteria.  Configurable via DecisionTreeConfig.  Thread-safe.
 */
template<typename T = double, typename = std::enable_if_t<IsNumericV<T>>>
class DecisionTree {
public:
    // ── Public type aliases (backward-compatible) ────────────────────────
    using Node     = DecisionTreeNode<T>;
    using NodePtr  = std::shared_ptr<Node>;

    /**
     * @brief Criterion enum — kept for backward compatibility.
     *
     * Maps 1:1 to SplitCriterionType from DecisionTreeConfig.hpp.
     */
    enum class Criterion {
        Gini    = static_cast<int>(SplitCriterionType::Gini),
        Entropy = static_cast<int>(SplitCriterionType::Entropy),
        MSE     = static_cast<int>(SplitCriterionType::MSE)
    };

    /**
     * @brief Backward-compatible Dataset alias.
     */
    using Dataset = Ml::Dataset<T>;

    /**
     * @brief Configuration struct for tree hyperparameters.
     *
     * Wraps DecisionTreeConfig<T> but exposes the legacy Criterion enum
     * so existing code keeps compiling without changes.
     */
    struct Config {
        Criterion SplitCriterion      = Criterion::Gini;
        size_t    MaxDepth            = 0;
        size_t    MinSamplesSplit     = 2;
        size_t    MinSamplesLeaf      = 1;
        size_t    MaxFeatures         = 0;
        T         MinImpurityDecrease = T{0};
        bool      RandomState        = false;
        unsigned int Seed             = 42;
    };

    // ── Construction ─────────────────────────────────────────────────────

    /**
     * @brief Construct a DecisionTree with the given configuration.
     * @param Cfg The tree configuration (uses defaults if omitted).
     */
    explicit DecisionTree(Config Cfg = Config())
        : Config_(std::move(Cfg)),
          RandomEngine_(Config_.Seed),
          Root_(nullptr) {}

    // ── Training ─────────────────────────────────────────────────────────

    /**
     * @brief Train the decision tree on the given dataset.
     * @param DatasetIn The training dataset (features + labels).
     * @throws std::invalid_argument if the dataset is invalid.
     */
    void Fit(const Dataset& DatasetIn) {
        std::lock_guard<std::mutex> Lock(Mutex_);

        if (!DatasetIn.Validate()) {
            throw std::invalid_argument(
                "Invalid dataset: dimensions mismatch or empty data");
        }

        NFeatures_ = DatasetIn.Features[0].size();

        if (!DatasetIn.FeatureNames.empty()) {
            FeatureNames_ = DatasetIn.FeatureNames;
        } else {
            FeatureNames_.resize(NFeatures_);
            for (size_t i = 0; i < NFeatures_; ++i) {
                std::stringstream Ss;
                Ss << "feature_" << i;
                FeatureNames_[i] = Ss.str();
            }
        }

        Classes_ = GetUniqueValues(DatasetIn.Labels);

        std::vector<size_t> Indices(DatasetIn.Features.size());
        std::iota(Indices.begin(), Indices.end(), 0);

        Root_ = BuildTree(DatasetIn, Indices, 0);
    }

    // ── Prediction ───────────────────────────────────────────────────────

    /**
     * @brief Predict the label for a single sample.
     * @param Features Feature vector for one sample.
     * @return The predicted label.
     */
    T Predict(const std::vector<T>& Features) const {
        std::lock_guard<std::mutex> Lock(Mutex_);

        if (!Root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }
        if (Features.size() != NFeatures_) {
            std::stringstream Ss;
            Ss << "Feature size mismatch. Expected " << NFeatures_
               << ", got " << Features.size();
            throw std::invalid_argument(Ss.str());
        }
        return Node::PredictSample(Features, Root_);
    }

    /**
     * @brief Predict labels for multiple samples.
     * @param Features Feature matrix [n_samples, n_features].
     * @return Vector of predicted labels.
     */
    std::vector<T> Predict(const std::vector<std::vector<T>>& Features) const {
        std::lock_guard<std::mutex> Lock(Mutex_);

        if (!Root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }

        std::vector<T> Predictions;
        Predictions.reserve(Features.size());

        for (const auto& Sample : Features) {
            if (Sample.size() != NFeatures_) {
                std::stringstream Ss;
                Ss << "Feature size mismatch at sample. Expected " << NFeatures_
                   << ", got " << Sample.size();
                throw std::invalid_argument(Ss.str());
            }
            Predictions.push_back(Node::PredictSample(Sample, Root_));
        }
        return Predictions;
    }

    // ── Inspection ───────────────────────────────────────────────────────

    /**
     * @brief Compute feature importance scores (normalized to sum to 1).
     * @return Vector of importance values, one per feature.
     */
    std::vector<double> FeatureImportance() const {
        std::lock_guard<std::mutex> Lock(Mutex_);

        if (!Root_) {
            throw std::runtime_error("Model not trained yet, call fit() first");
        }

        std::vector<double> Importance(NFeatures_, 0.0);
        Node::ComputeFeatureImportance(Root_, Importance);

        double Total = std::accumulate(Importance.begin(), Importance.end(), 0.0);
        if (Total > 0) {
            for (auto& v : Importance) v /= Total;
        }
        return Importance;
    }

    /**
     * @brief Get the maximum depth of the trained tree.
     */
    size_t GetDepth() const {
        std::lock_guard<std::mutex> Lock(Mutex_);
        if (!Root_) return 0;
        return Node::ComputeDepth(Root_);
    }

    /**
     * @brief Get the total number of nodes in the tree.
     */
    size_t GetNodeCount() const {
        std::lock_guard<std::mutex> Lock(Mutex_);
        if (!Root_) return 0;
        return Node::CountNodes(Root_);
    }

    /**
     * @brief Print the tree structure to stdout (for debugging).
     */
    void PrintTree() const {
        std::lock_guard<std::mutex> Lock(Mutex_);
        if (!Root_) {
            std::cout << "Tree not trained yet" << std::endl;
            return;
        }
        Node::PrintNode(Root_, FeatureNames_, 0);
    }

private:
    Config                   Config_;
    mutable std::mutex       Mutex_;
    mutable std::mt19937     RandomEngine_;
    size_t                   NFeatures_ = 0;
    std::vector<T>           Classes_;
    std::vector<std::string> FeatureNames_;
    NodePtr                  Root_;

    // ── Utility helpers ──────────────────────────────────────────────────

    static std::vector<T> GetUniqueValues(const std::vector<T>& Values) {
        std::vector<T> unique = Values;
        std::sort(unique.begin(), unique.end());
        auto Last = std::unique(unique.begin(), unique.end());
        unique.erase(Last, unique.end());
        return unique;
    }

    static std::unordered_map<T, size_t> CountValues(
        const std::vector<T>& Values,
        const std::vector<size_t>& Indices) {

        std::unordered_map<T, size_t> Counts;
        for (size_t Idx : Indices) {
            Counts[Values[Idx]]++;
        }
        return Counts;
    }

    // ── Impurity calculations ────────────────────────────────────────────

    static double CalculateEntropy(const std::unordered_map<T, size_t>& ClassCounts,
                                   size_t Total) {
        double Entropy = 0.0;
        for (const auto& pair : ClassCounts) {
            double p = static_cast<double>(pair.second) / Total;
            if (p > 0) Entropy -= p * std::log2(p);
        }
        return Entropy;
    }

    static double CalculateGini(const std::unordered_map<T, size_t>& ClassCounts,
                                size_t Total) {
        double Gini = 1.0;
        for (const auto& pair : ClassCounts) {
            double p = static_cast<double>(pair.second) / Total;
            Gini -= p * p;
        }
        return Gini;
    }

    static double CalculateMse(const std::vector<T>& Values,
                               const std::vector<size_t>& Indices) {
        if (Indices.empty()) return 0.0;
        T Sum = 0;
        for (size_t Idx : Indices) Sum += Values[Idx];
        T Mean = Sum / static_cast<T>(Indices.size());
        double Mse = 0.0;
        for (size_t Idx : Indices) {
            double Diff = static_cast<double>(Values[Idx]) - static_cast<double>(Mean);
            Mse += Diff * Diff;
        }
        return Mse / Indices.size();
    }

    static T GetPredictionValue(const std::vector<T>& Values,
                                const std::vector<size_t>& Indices,
                                Criterion SplitCriterion) {
        if (SplitCriterion == Criterion::MSE) {
            T Sum = 0;
            for (size_t Idx : Indices) Sum += Values[Idx];
            return Sum / static_cast<T>(Indices.size());
        }
        // Classification: majority vote
        std::unordered_map<T, size_t> Counts;
        for (size_t Idx : Indices) Counts[Values[Idx]]++;
        T MostCommon{};
        size_t MaxCount = 0;
        for (const auto& pair : Counts) {
            if (pair.second > MaxCount) {
                MostCommon = pair.first;
                MaxCount   = pair.second;
            }
        }
        return MostCommon;
    }

    double CalculateImpurity(const std::vector<T>& Values,
                             const std::vector<size_t>& Indices) const {
        if (Indices.empty()) return 0.0;
        switch (Config_.SplitCriterion) {
            case Criterion::Gini: {
                auto Counts = CountValues(Values, Indices);
                return CalculateGini(Counts, Indices.size());
            }
            case Criterion::Entropy: {
                auto Counts = CountValues(Values, Indices);
                return CalculateEntropy(Counts, Indices.size());
            }
            case Criterion::MSE:
                return CalculateMse(Values, Indices);
            default:
                throw std::invalid_argument("Unknown criterion");
        }
    }

    // ── Tree building ────────────────────────────────────────────────────

    std::optional<std::tuple<size_t, T, double,
                             std::vector<size_t>, std::vector<size_t>>>
    FindBestSplit(const Dataset& DatasetIn,
                  const std::vector<size_t>& Indices) const {

        if (Indices.size() < Config_.MinSamplesSplit) {
            return std::nullopt;
        }

        double BestGain = 0.0;
        size_t BestFeature = 0;
        T BestThreshold = T{};
        std::vector<size_t> BestLeftIndices;
        std::vector<size_t> BestRightIndices;

        double CurrentImpurity = CalculateImpurity(DatasetIn.Labels, Indices);

        std::vector<size_t> FeatureIndices(NFeatures_);
        std::iota(FeatureIndices.begin(), FeatureIndices.end(), 0);

        if (Config_.MaxFeatures > 0 && Config_.MaxFeatures < NFeatures_) {
            std::mt19937 TempEngine = RandomEngine_;
            std::shuffle(FeatureIndices.begin(), FeatureIndices.end(), TempEngine);
            FeatureIndices.resize(Config_.MaxFeatures);
        }

        for (size_t FeatureIdx : FeatureIndices) {
            std::vector<T> FeatureValues;
            FeatureValues.reserve(Indices.size());
            for (size_t Idx : Indices) {
                FeatureValues.push_back(DatasetIn.Features[Idx][FeatureIdx]);
            }
            std::sort(FeatureValues.begin(), FeatureValues.end());
            auto Last = std::unique(FeatureValues.begin(), FeatureValues.end());
            FeatureValues.erase(Last, FeatureValues.end());

            if (FeatureValues.size() < 2) continue;

            for (size_t i = 0; i < FeatureValues.size() - 1; ++i) {
                T threshold = (FeatureValues[i] + FeatureValues[i + 1]) / 2;

                std::vector<size_t> LeftIndices;
                std::vector<size_t> RightIndices;

                for (size_t Idx : Indices) {
                    if (DatasetIn.Features[Idx][FeatureIdx] <= threshold) {
                        LeftIndices.push_back(Idx);
                    } else {
                        RightIndices.push_back(Idx);
                    }
                }

                if (LeftIndices.size() < Config_.MinSamplesLeaf ||
                    RightIndices.size() < Config_.MinSamplesLeaf) {
                    continue;
                }

                double LeftImpurity  = CalculateImpurity(DatasetIn.Labels, LeftIndices);
                double RightImpurity = CalculateImpurity(DatasetIn.Labels, RightIndices);

                double n = static_cast<double>(Indices.size());
                double WeightedImpurity =
                    (LeftIndices.size() / n) * LeftImpurity +
                    (RightIndices.size() / n) * RightImpurity;

                double Gain = CurrentImpurity - WeightedImpurity;

                if (Gain > BestGain && Gain >= Config_.MinImpurityDecrease) {
                    BestGain         = Gain;
                    BestFeature      = FeatureIdx;
                    BestThreshold    = threshold;
                    BestLeftIndices  = std::move(LeftIndices);
                    BestRightIndices = std::move(RightIndices);
                }
            }
        }

        if (BestGain > 0) {
            return std::make_tuple(BestFeature, BestThreshold, BestGain,
                                   std::move(BestLeftIndices),
                                   std::move(BestRightIndices));
        }
        return std::nullopt;
    }

    NodePtr BuildTree(const Dataset& DatasetIn,
                      const std::vector<size_t>& Indices,
                      size_t Depth) {

        if (Indices.empty()) return nullptr;

        double CurrentImpurity = CalculateImpurity(DatasetIn.Labels, Indices);
        T NodeValue = GetPredictionValue(DatasetIn.Labels, Indices,
                                         Config_.SplitCriterion);

        if ((Config_.MaxDepth > 0 && Depth >= Config_.MaxDepth) ||
            Indices.size() < Config_.MinSamplesSplit) {
            return std::make_shared<Node>(NodeValue, CurrentImpurity,
                                          Indices.size());
        }

        auto SplitResult = FindBestSplit(DatasetIn, Indices);
        if (!SplitResult) {
            return std::make_shared<Node>(NodeValue, CurrentImpurity,
                                          Indices.size());
        }

        auto [FeatureIdx, threshold, Gain, LeftIndices, RightIndices] = *SplitResult;

        auto NewNode = std::make_shared<Node>(FeatureIdx, threshold,
                                              CurrentImpurity, Indices.size());
        NewNode->Left  = BuildTree(DatasetIn, LeftIndices,  Depth + 1);
        NewNode->Right = BuildTree(DatasetIn, RightIndices, Depth + 1);
        return NewNode;
    }
};

} // namespace Ml
