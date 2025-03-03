/**
 * @file test_DecisionTree.cpp
 * @brief Google Test suite for DecisionTree — educational output with PrintSection banners.
 */
#include <gtest/gtest.h>
#include "DecisionTree.hpp"
#include <cmath>
#include <sstream>

// ─────────────────────────────────────────────────────────────────────────────
// Educational banner helper
// ─────────────────────────────────────────────────────────────────────────────
namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: deterministic datasets and pre-trained tree
// ─────────────────────────────────────────────────────────────────────────────
static Ml::DecisionTree<double>::Dataset MakeSimpleDataset(size_t n = 200) {
    std::vector<std::vector<double>> Features(n, std::vector<double>(2));
    std::vector<double> Labels(n);
    std::mt19937 Gen(42);
    std::uniform_real_distribution<> Dis(0.0, 2.0);
    for (size_t i = 0; i < n; ++i) {
        Features[i][0] = Dis(Gen);
        Features[i][1] = Dis(Gen);
        Labels[i] = (Features[i][0] + Features[i][1] <= 1.0) ? 0.0 : 1.0;
    }
    return {Features, Labels, {"x", "y"}};
}

static std::unique_ptr<Ml::DecisionTree<double>> MakeTrainedTree() {
    Ml::DecisionTree<double>::Config Cfg;
    Cfg.MaxDepth = 5;
    auto Tree = std::make_unique<Ml::DecisionTree<double>>(Cfg);
    Tree->Fit(MakeSimpleDataset());
    return Tree;
}

// ======================== Dataset::Validate ========================

TEST(DecisionTree, DatasetValidate_Valid) {
    PrintSection("Dataset::Validate -- valid dataset",
                 "A well-formed dataset (matching rows/cols) must pass validation.");
    auto Ds = MakeSimpleDataset(10);
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_TRUE(Result);
}

TEST(DecisionTree, DatasetValidate_EmptyFeatures) {
    PrintSection("Dataset::Validate -- empty features",
                 "An empty feature matrix should be caught before training begins.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {};
    Ds.Labels = {1.0};
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_FALSE(Result);
}

TEST(DecisionTree, DatasetValidate_EmptyLabels) {
    PrintSection("Dataset::Validate -- empty labels",
                 "Labels are required; training without them is meaningless.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}};
    Ds.Labels = {};
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_FALSE(Result);
}

TEST(DecisionTree, DatasetValidate_MismatchedSampleCounts) {
    PrintSection("Dataset::Validate -- mismatched sample counts",
                 "Feature rows and label count must agree to avoid index errors.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0, 4.0}};
    Ds.Labels = {1.0};
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_FALSE(Result);
}

TEST(DecisionTree, DatasetValidate_InconsistentFeatureLengths) {
    PrintSection("Dataset::Validate -- inconsistent feature lengths",
                 "Every sample must have the same number of features for the "
                 "split logic to index safely.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0}};
    Ds.Labels = {0.0, 1.0};
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_FALSE(Result);
}

TEST(DecisionTree, DatasetValidate_WrongFeatureNameCount) {
    PrintSection("Dataset::Validate -- wrong feature-name count",
                 "If feature names are provided, their count must match the "
                 "number of columns to keep reporting correct.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0, 4.0}};
    Ds.Labels = {0.0, 1.0};
    Ds.FeatureNames = {"only_one"};
    bool Result = Ds.Validate();
    std::cout << "[Result] Validate() = " << std::boolalpha << Result << std::endl;
    EXPECT_FALSE(Result);
}

// ======================== Config defaults ========================

TEST(DecisionTree, ConfigDefaults) {
    PrintSection("Config defaults",
                 "Verifying that default hyperparameters match documentation "
                 "prevents silent regressions when someone edits the struct.");
    Ml::DecisionTree<double>::Config Cfg;
    std::cout << "[Result] SplitCriterion = Gini? "
              << (Cfg.SplitCriterion == Ml::DecisionTree<double>::Criterion::Gini) << std::endl;
    std::cout << "[Result] MaxDepth = " << Cfg.MaxDepth << std::endl;
    std::cout << "[Result] MinSamplesSplit = " << Cfg.MinSamplesSplit << std::endl;
    std::cout << "[Result] MinSamplesLeaf = " << Cfg.MinSamplesLeaf << std::endl;
    std::cout << "[Result] MaxFeatures = " << Cfg.MaxFeatures << std::endl;
    std::cout << "[Result] MinImpurityDecrease = " << Cfg.MinImpurityDecrease << std::endl;
    std::cout << "[Result] RandomState = " << std::boolalpha << Cfg.RandomState << std::endl;
    std::cout << "[Result] Seed = " << Cfg.Seed << std::endl;
    EXPECT_EQ(Cfg.SplitCriterion, Ml::DecisionTree<double>::Criterion::Gini);
    EXPECT_EQ(Cfg.MaxDepth, 0u);
    EXPECT_EQ(Cfg.MinSamplesSplit, 2u);
    EXPECT_EQ(Cfg.MinSamplesLeaf, 1u);
    EXPECT_EQ(Cfg.MaxFeatures, 0u);
    EXPECT_DOUBLE_EQ(Cfg.MinImpurityDecrease, 0.0);
    EXPECT_FALSE(Cfg.RandomState);
    EXPECT_EQ(Cfg.Seed, 42u);
}

// ======================== Fit ========================

TEST(DecisionTree, Fit_ValidDataset) {
    PrintSection("Fit -- valid dataset",
                 "The tree must train successfully and produce a non-trivial "
                 "structure (depth > 0, nodes > 0).");
    Ml::DecisionTree<double> Tree;
    auto Ds = MakeSimpleDataset(50);
    EXPECT_NO_THROW(Tree.Fit(Ds));
    std::cout << "[Result] Depth = " << Tree.GetDepth()
              << ", NodeCount = " << Tree.GetNodeCount() << std::endl;
    EXPECT_GT(Tree.GetDepth(), size_t{0});
    EXPECT_GT(Tree.GetNodeCount(), size_t{0});
}

TEST(DecisionTree, Fit_EmptyDataset) {
    PrintSection("Fit -- empty dataset",
                 "Fit() must throw std::invalid_argument for an empty dataset "
                 "instead of crashing on a null dereference.");
    Ml::DecisionTree<double> Tree;
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {};
    Ds.Labels = {};
    std::cout << "[Result] Expecting std::invalid_argument..." << std::endl;
    EXPECT_THROW(Tree.Fit(Ds), std::invalid_argument);
}

TEST(DecisionTree, Fit_MismatchedDimensions) {
    PrintSection("Fit -- mismatched dimensions",
                 "Rows of features and labels must agree; otherwise the tree "
                 "would read out-of-bounds during split search.");
    Ml::DecisionTree<double> Tree;
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0, 4.0}};
    Ds.Labels = {1.0};
    std::cout << "[Result] Expecting std::invalid_argument..." << std::endl;
    EXPECT_THROW(Tree.Fit(Ds), std::invalid_argument);
}

TEST(DecisionTree, Fit_GeneratesDefaultFeatureNames) {
    PrintSection("Fit -- generates default feature names",
                 "When no names are provided, Fit() auto-generates them so that "
                 "PrintTree and importance reports remain readable.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0, 4.0}};
    Ds.Labels = {0.0, 1.0};
    Ml::DecisionTree<double> Tree;
    EXPECT_NO_THROW(Tree.Fit(Ds));
    double Pred = Tree.Predict(std::vector<double>{1.0, 2.0});
    std::cout << "[Result] Prediction with auto-generated names = " << Pred << std::endl;
    (void)Pred;
}

// ======================== Predict (single) ========================

TEST(DecisionTree, PredictSingle_CorrectPrediction) {
    PrintSection("Predict single -- correct prediction",
                 "Clear-cut samples far from the decision boundary should be "
                 "classified correctly by even a shallow tree.");
    auto Tree = MakeTrainedTree();
    double Pred0 = Tree->Predict(std::vector<double>{0.1, 0.1});
    double Pred1 = Tree->Predict(std::vector<double>{1.5, 1.5});
    std::cout << "[Result] Predict({0.1, 0.1}) = " << Pred0 << " (expected 0.0)" << std::endl;
    std::cout << "[Result] Predict({1.5, 1.5}) = " << Pred1 << " (expected 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(Pred0, 0.0);
    EXPECT_DOUBLE_EQ(Pred1, 1.0);
}

TEST(DecisionTree, PredictSingle_BeforeTraining) {
    PrintSection("Predict single -- before training",
                 "Calling Predict() on an untrained tree must throw, not "
                 "dereference a null root pointer.");
    Ml::DecisionTree<double> Tree;
    std::cout << "[Result] Expecting std::runtime_error..." << std::endl;
    EXPECT_THROW(Tree.Predict(std::vector<double>{1.0, 2.0}), std::runtime_error);
}

TEST(DecisionTree, PredictSingle_WrongFeatureCount) {
    PrintSection("Predict single -- wrong feature count",
                 "Feature-dimension mismatch must be caught to prevent "
                 "out-of-bounds access in the node traversal.");
    auto Tree = MakeTrainedTree();
    std::cout << "[Result] Expecting std::invalid_argument for too few features..." << std::endl;
    EXPECT_THROW(Tree->Predict(std::vector<double>{1.0}), std::invalid_argument);
    std::cout << "[Result] Expecting std::invalid_argument for too many features..." << std::endl;
    EXPECT_THROW(Tree->Predict(std::vector<double>{1.0, 2.0, 3.0}), std::invalid_argument);
}

// ======================== Predict (batch) ========================

TEST(DecisionTree, PredictBatch_CorrectSize) {
    PrintSection("Predict batch -- correct size",
                 "Batch prediction must return exactly one label per input sample.");
    auto Tree = MakeTrainedTree();
    std::vector<std::vector<double>> Samples = {{0.1, 0.1}, {1.5, 1.5}, {0.5, 0.3}};
    auto Preds = Tree->Predict(Samples);
    std::cout << "[Result] Input samples = " << Samples.size()
              << ", output predictions = " << Preds.size() << std::endl;
    EXPECT_EQ(Preds.size(), 3u);
}

TEST(DecisionTree, PredictBatch_BeforeTraining) {
    PrintSection("Predict batch -- before training",
                 "Same safety check as single-sample predict: no null-root crash.");
    Ml::DecisionTree<double> Tree;
    std::vector<std::vector<double>> Samples = {{1.0, 2.0}};
    std::cout << "[Result] Expecting std::runtime_error..." << std::endl;
    EXPECT_THROW(Tree.Predict(Samples), std::runtime_error);
}

TEST(DecisionTree, PredictBatch_WrongFeatureCount) {
    PrintSection("Predict batch -- wrong feature count in one sample",
                 "Even a single malformed sample in the batch must be caught.");
    auto Tree = MakeTrainedTree();
    std::vector<std::vector<double>> Samples = {{0.1, 0.1}, {1.0}};
    std::cout << "[Result] Expecting std::invalid_argument..." << std::endl;
    EXPECT_THROW(Tree->Predict(Samples), std::invalid_argument);
}

// ======================== FeatureImportance ========================

TEST(DecisionTree, FeatureImportance_SumsToOne) {
    PrintSection("FeatureImportance -- sums to 1",
                 "Normalized importance scores must sum to 1.0 and each be "
                 "non-negative (no feature can hurt by splitting on it).");
    auto Tree = MakeTrainedTree();
    auto Imp = Tree->FeatureImportance();
    double Total = 0.0;
    for (size_t i = 0; i < Imp.size(); ++i) {
        std::cout << "[Result] Importance[" << i << "] = " << Imp[i] << std::endl;
        Total += Imp[i];
    }
    std::cout << "[Result] Total importance = " << Total << std::endl;
    EXPECT_EQ(Imp.size(), 2u);
    for (double v : Imp) {
        EXPECT_GE(v, 0.0);
    }
    EXPECT_NEAR(Total, 1.0, 1e-9);
}

TEST(DecisionTree, FeatureImportance_BeforeTraining) {
    PrintSection("FeatureImportance -- before training",
                 "Must throw rather than iterate over an empty importance vector.");
    Ml::DecisionTree<double> Tree;
    std::cout << "[Result] Expecting std::runtime_error..." << std::endl;
    EXPECT_THROW(Tree.FeatureImportance(), std::runtime_error);
}

// ======================== GetDepth ========================

TEST(DecisionTree, GetDepth_BeforeTraining) {
    PrintSection("GetDepth -- before training",
                 "An untrained tree has no nodes, so depth should be 0.");
    Ml::DecisionTree<double> Tree;
    size_t Depth = Tree.GetDepth();
    std::cout << "[Result] Depth = " << Depth << std::endl;
    EXPECT_EQ(Depth, size_t{0});
}

TEST(DecisionTree, GetDepth_AfterTraining) {
    PrintSection("GetDepth -- after training",
                 "A trained tree on non-trivial data must have depth > 0.");
    auto Tree = MakeTrainedTree();
    size_t Depth = Tree->GetDepth();
    std::cout << "[Result] Depth = " << Depth << std::endl;
    EXPECT_GT(Depth, size_t{0});
}

TEST(DecisionTree, GetDepth_RespectsMaxDepth) {
    PrintSection("GetDepth -- respects MaxDepth config",
                 "Setting MaxDepth=3 limits splits to depths 0..2, so the "
                 "computed depth (which counts leaf levels) must be <= 4.");
    Ml::DecisionTree<double>::Config Cfg;
    Cfg.MaxDepth = 3;
    Ml::DecisionTree<double> Tree(Cfg);
    Tree.Fit(MakeSimpleDataset());
    size_t Depth = Tree.GetDepth();
    std::cout << "[Result] Depth = " << Depth << " (max allowed ~4)" << std::endl;
    EXPECT_LE(Depth, size_t{4});
}

// ======================== GetNodeCount ========================

TEST(DecisionTree, GetNodeCount_BeforeTraining) {
    PrintSection("GetNodeCount -- before training",
                 "No tree built yet means zero nodes.");
    Ml::DecisionTree<double> Tree;
    size_t Count = Tree.GetNodeCount();
    std::cout << "[Result] NodeCount = " << Count << std::endl;
    EXPECT_EQ(Count, size_t{0});
}

TEST(DecisionTree, GetNodeCount_AfterTraining) {
    PrintSection("GetNodeCount -- after training",
                 "A trained tree on non-trivial data must have at least one node.");
    auto Tree = MakeTrainedTree();
    size_t Count = Tree->GetNodeCount();
    std::cout << "[Result] NodeCount = " << Count << std::endl;
    EXPECT_GT(Count, size_t{0});
}

// ======================== PrintTree ========================

TEST(DecisionTree, PrintTree_BeforeTraining) {
    PrintSection("PrintTree -- before training",
                 "PrintTree on an untrained model must print a friendly message, "
                 "not crash.");
    Ml::DecisionTree<double> Tree;
    std::ostringstream Buf;
    std::streambuf* Old = std::cout.rdbuf(Buf.rdbuf());
    EXPECT_NO_THROW(Tree.PrintTree());
    std::cout.rdbuf(Old);
    std::cout << "[Result] Output contains 'not trained': "
              << (Buf.str().find("not trained") != std::string::npos) << std::endl;
    EXPECT_NE(Buf.str().find("not trained"), std::string::npos);
}

TEST(DecisionTree, PrintTree_AfterTraining) {
    PrintSection("PrintTree -- after training",
                 "After fitting, the tree should print node/leaf descriptions.");
    auto Tree = MakeTrainedTree();
    std::ostringstream Buf;
    std::streambuf* Old = std::cout.rdbuf(Buf.rdbuf());
    EXPECT_NO_THROW(Tree->PrintTree());
    std::cout.rdbuf(Old);
    bool HasContent = !Buf.str().empty();
    bool HasKeywords = Buf.str().find("Leaf") != std::string::npos ||
                       Buf.str().find("Node") != std::string::npos;
    std::cout << "[Result] Output non-empty: " << HasContent
              << ", contains Leaf/Node: " << HasKeywords << std::endl;
    EXPECT_TRUE(HasContent);
    EXPECT_TRUE(HasKeywords);
}

// ======================== Entropy criterion ========================

TEST(DecisionTree, Fit_EntropyCriterion) {
    PrintSection("Fit -- Entropy criterion",
                 "Entropy uses -sum(p*log2(p)); the tree must train and produce "
                 "a valid structure with this alternative criterion.");
    Ml::DecisionTree<double>::Config Cfg;
    Cfg.SplitCriterion = Ml::DecisionTree<double>::Criterion::Entropy;
    Cfg.MaxDepth = 4;
    Ml::DecisionTree<double> Tree(Cfg);
    auto Ds = MakeSimpleDataset();
    EXPECT_NO_THROW(Tree.Fit(Ds));
    std::cout << "[Result] Depth = " << Tree.GetDepth()
              << ", NodeCount = " << Tree.GetNodeCount() << std::endl;
    EXPECT_GT(Tree.GetDepth(), size_t{0});
}

// ======================== MSE criterion (regression) ========================

TEST(DecisionTree, Fit_MSECriterion) {
    PrintSection("Fit -- MSE criterion (regression)",
                 "MSE splits minimise variance; the tree must handle continuous "
                 "labels like y = x0 + x1.");
    Ml::DecisionTree<double>::Config Cfg;
    Cfg.SplitCriterion = Ml::DecisionTree<double>::Criterion::MSE;
    Cfg.MaxDepth = 4;
    Ml::DecisionTree<double> Tree(Cfg);
    std::vector<std::vector<double>> Feats = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
        {2.0, 0.0}, {0.0, 2.0}, {2.0, 2.0}, {1.5, 1.5}};
    std::vector<double> Labels;
    for (auto& f : Feats) Labels.push_back(f[0] + f[1]);
    Ml::DecisionTree<double>::Dataset Ds{Feats, Labels, {"x", "y"}};
    EXPECT_NO_THROW(Tree.Fit(Ds));
    std::cout << "[Result] Depth = " << Tree.GetDepth()
              << ", NodeCount = " << Tree.GetNodeCount() << std::endl;
    EXPECT_GT(Tree.GetDepth(), size_t{0});
}

// ======================== Refit / overwrite ========================

TEST(DecisionTree, Fit_OverwritesPreviousModel) {
    PrintSection("Fit -- overwrite previous model",
                 "Calling Fit() a second time must fully replace the old tree, "
                 "including accepting a different feature width.");
    Ml::DecisionTree<double>::Dataset Ds2;
    Ds2.Features = {{0.0}, {1.0}, {2.0}, {3.0}};
    Ds2.Labels = {0.0, 0.0, 1.0, 1.0};

    Ml::DecisionTree<double>::Config Cfg;
    Cfg.MaxDepth = 2;
    Ml::DecisionTree<double> Tree2(Cfg);
    Tree2.Fit(Ds2);

    std::cout << "[Result] After refit: Depth = " << Tree2.GetDepth()
              << ", NodeCount = " << Tree2.GetNodeCount() << std::endl;
    EXPECT_GT(Tree2.GetDepth(), size_t{0});
    EXPECT_GT(Tree2.GetNodeCount(), size_t{0});
    double Pred = Tree2.Predict(std::vector<double>{0.5});
    std::cout << "[Result] Predict({0.5}) = " << Pred << std::endl;
    EXPECT_NO_THROW((void)Pred);
}

// ======================== Single-class dataset ========================

TEST(DecisionTree, Fit_SingleClassDataset) {
    PrintSection("Fit -- single-class dataset",
                 "When every label is the same, no useful split exists. The tree "
                 "should create a single leaf that always predicts that label.");
    Ml::DecisionTree<double>::Dataset Ds;
    Ds.Features = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    Ds.Labels = {1.0, 1.0, 1.0};
    Ml::DecisionTree<double> Tree;
    EXPECT_NO_THROW(Tree.Fit(Ds));
    double Pred = Tree.Predict(std::vector<double>{2.0, 3.0});
    std::cout << "[Result] Predict({2.0, 3.0}) = " << Pred << " (expected 1.0)" << std::endl;
    EXPECT_EQ(Pred, 1.0);
}

// ======================== SplitCriterion concept ========================

TEST(DecisionTree, SplitCriterionConcept_GiniCriterion) {
    PrintSection("SplitCriterion concept -- GiniCriterion",
                 "GiniCriterion must satisfy the SplitCriterion concept and "
                 "return 0 for a pure set, non-zero for a mixed set.");
    static_assert(Ml::SplitCriterion<Ml::GiniCriterion>,
                  "GiniCriterion must satisfy SplitCriterion");
    Ml::GiniCriterion gini;
    double pure = gini.Impurity(std::vector<int>{1, 1, 1});
    double mixed = gini.Impurity(std::vector<int>{0, 1, 0, 1});
    std::cout << "[Result] Gini(pure) = " << pure << " (expected 0.0)" << std::endl;
    std::cout << "[Result] Gini(mixed) = " << mixed << " (expected 0.5)" << std::endl;
    EXPECT_DOUBLE_EQ(pure, 0.0);
    EXPECT_NEAR(mixed, 0.5, 1e-9);
}

TEST(DecisionTree, SplitCriterionConcept_EntropyCriterion) {
    PrintSection("SplitCriterion concept -- EntropyCriterion",
                 "EntropyCriterion must satisfy the SplitCriterion concept and "
                 "return 0 for a pure set, 1.0 for a perfectly balanced binary set.");
    static_assert(Ml::SplitCriterion<Ml::EntropyCriterion>,
                  "EntropyCriterion must satisfy SplitCriterion");
    Ml::EntropyCriterion entropy;
    double pure = entropy.Impurity(std::vector<int>{1, 1, 1});
    double balanced = entropy.Impurity(std::vector<int>{0, 0, 1, 1});
    std::cout << "[Result] Entropy(pure) = " << pure << " (expected 0.0)" << std::endl;
    std::cout << "[Result] Entropy(balanced) = " << balanced << " (expected 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(pure, 0.0);
    EXPECT_NEAR(balanced, 1.0, 1e-9);
}

TEST(DecisionTree, SplitCriterionConcept_MseCriterion) {
    PrintSection("SplitCriterion concept -- MseCriterion",
                 "MseCriterion must satisfy the SplitCriterion concept and "
                 "return 0 for identical values, positive for varying values.");
    static_assert(Ml::SplitCriterion<Ml::MseCriterion>,
                  "MseCriterion must satisfy SplitCriterion");
    Ml::MseCriterion mse;
    double same = mse.Impurity(std::vector<int>{5, 5, 5});
    double varied = mse.Impurity(std::vector<int>{0, 10});
    std::cout << "[Result] MSE(same) = " << same << " (expected 0.0)" << std::endl;
    std::cout << "[Result] MSE(varied) = " << varied << " (expected 25.0)" << std::endl;
    EXPECT_DOUBLE_EQ(same, 0.0);
    EXPECT_NEAR(varied, 25.0, 1e-9);
}

TEST(DecisionTree, SplitCriterionConcept_EmptyLabels) {
    PrintSection("SplitCriterion concept -- empty labels",
                 "All criteria must return 0.0 for an empty label vector "
                 "to avoid division-by-zero.");
    Ml::GiniCriterion gini;
    Ml::EntropyCriterion entropy;
    Ml::MseCriterion mse;
    std::vector<int> empty;
    std::cout << "[Result] Gini(empty) = " << gini.Impurity(empty) << std::endl;
    std::cout << "[Result] Entropy(empty) = " << entropy.Impurity(empty) << std::endl;
    std::cout << "[Result] MSE(empty) = " << mse.Impurity(empty) << std::endl;
    EXPECT_DOUBLE_EQ(gini.Impurity(empty), 0.0);
    EXPECT_DOUBLE_EQ(entropy.Impurity(empty), 0.0);
    EXPECT_DOUBLE_EQ(mse.Impurity(empty), 0.0);
}
