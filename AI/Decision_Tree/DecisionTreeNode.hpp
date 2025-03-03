/**
 * @file DecisionTreeNode.hpp
 * @brief Node structure and Dataset definition for the DecisionTree.
 *
 * @details Defines:
 *   - Dataset: feature matrix + label vector with validation.
 *   - Node: internal tree node (decision or leaf) with traversal helpers.
 */
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <type_traits>

namespace Ml {

// ─────────────────────────────────────────────────────────────────────────────
// Dataset
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Holds a feature matrix, label vector, and optional feature names.
 * @tparam T Numeric type for features and labels.
 */
template<typename T>
struct Dataset {
    std::vector<std::vector<T>> Features;   ///< [n_samples, n_features]
    std::vector<T> Labels;                  ///< [n_samples]
    std::vector<std::string> FeatureNames;  ///< Optional; must match n_features if set.

    /**
     * @brief Validate that the dataset dimensions are consistent.
     * @return true if the dataset is valid.
     */
    bool Validate() const {
        if (Features.empty() || Labels.empty()) {
            return false;
        }
        size_t NSamples = Features.size();
        if (Labels.size() != NSamples) {
            return false;
        }
        size_t NFeatures = Features[0].size();
        for (const auto& Sample : Features) {
            if (Sample.size() != NFeatures) {
                return false;
            }
        }
        if (!FeatureNames.empty() && FeatureNames.size() != NFeatures) {
            return false;
        }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Node
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief A single node in the decision tree (leaf or internal split).
 * @tparam T Numeric type for thresholds and prediction values.
 */
template<typename T>
struct DecisionTreeNode {
    bool   IsLeaf       = false;
    T      Value        = T{};
    size_t FeatureIndex = 0;
    T      Threshold    = T{};
    double Impurity     = 0.0;
    size_t NSamples     = 0;

    std::shared_ptr<DecisionTreeNode> Left;
    std::shared_ptr<DecisionTreeNode> Right;

    /// Construct a leaf node.
    DecisionTreeNode(T PredictionValue, double NodeImpurity, size_t Samples)
        : IsLeaf(true), Value(PredictionValue),
          Impurity(NodeImpurity), NSamples(Samples) {}

    /// Construct a decision (internal) node.
    DecisionTreeNode(size_t FeatIndex, T SplitThreshold,
                     double NodeImpurity, size_t Samples)
        : IsLeaf(false), FeatureIndex(FeatIndex), Threshold(SplitThreshold),
          Impurity(NodeImpurity), NSamples(Samples) {}

    // ── Traversal helpers ───────────────────────────────────────────────

    /**
     * @brief Walk the tree to predict a label for a single sample.
     * @param features Feature vector for one sample.
     * @return The predicted value at the reached leaf.
     */
    static T PredictSample(const std::vector<T>& Features,
                           const std::shared_ptr<DecisionTreeNode>& Node) {
        if (!Node) {
            throw std::runtime_error("Invalid tree node");
        }
        if (Node->IsLeaf) {
            return Node->Value;
        }
        if (Features[Node->FeatureIndex] <= Node->Threshold) {
            return PredictSample(Features, Node->Left);
        }
        return PredictSample(Features, Node->Right);
    }

    /**
     * @brief Compute depth of the subtree rooted at this node.
     */
    static size_t ComputeDepth(const std::shared_ptr<DecisionTreeNode>& Node) {
        if (!Node) return 0;
        if (Node->IsLeaf) return 1;
        return 1 + std::max(ComputeDepth(Node->Left),
                            ComputeDepth(Node->Right));
    }

    /**
     * @brief Count total nodes in the subtree rooted at this node.
     */
    static size_t CountNodes(const std::shared_ptr<DecisionTreeNode>& Node) {
        if (!Node) return 0;
        return 1 + CountNodes(Node->Left) + CountNodes(Node->Right);
    }

    /**
     * @brief Accumulate feature importance from impurity decrease.
     */
    static void ComputeFeatureImportance(
        const std::shared_ptr<DecisionTreeNode>& Node,
        std::vector<double>& Importance) {

        if (!Node || Node->IsLeaf) return;

        double NodeImp = Node->NSamples * Node->Impurity;
        if (Node->Left) {
            NodeImp -= Node->Left->NSamples * Node->Left->Impurity;
        }
        if (Node->Right) {
            NodeImp -= Node->Right->NSamples * Node->Right->Impurity;
        }
        Importance[Node->FeatureIndex] += NodeImp;

        ComputeFeatureImportance(Node->Left, Importance);
        ComputeFeatureImportance(Node->Right, Importance);
    }

    /**
     * @brief Pretty-print the subtree (for debugging).
     */
    static void PrintNode(const std::shared_ptr<DecisionTreeNode>& Node,
                          const std::vector<std::string>& FeatureNames,
                          size_t Depth = 0) {
        if (!Node) return;

        std::string Indent(Depth * 4, ' ');

        if (Node->IsLeaf) {
            std::cout << Indent << "Leaf: value=" << Node->Value
                      << ", samples=" << Node->NSamples
                      << ", impurity=" << Node->Impurity << std::endl;
        } else {
            std::cout << Indent << "Node: feature=" << FeatureNames[Node->FeatureIndex]
                      << ", threshold=" << Node->Threshold
                      << ", samples=" << Node->NSamples
                      << ", impurity=" << Node->Impurity << std::endl;
            PrintNode(Node->Left, FeatureNames, Depth + 1);
            PrintNode(Node->Right, FeatureNames, Depth + 1);
        }
    }
};

} // namespace Ml
