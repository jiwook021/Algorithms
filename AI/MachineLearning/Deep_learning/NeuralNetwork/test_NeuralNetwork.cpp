/**
 * @file test_NeuralNetwork.cpp
 * @brief Unit tests for the NeuralNetwork module.
 *
 * Educational test output: each test prints a section banner explaining *what*
 * is being tested and *why*, followed by [Result] lines showing computed vs
 * expected values before the EXPECT macros.
 */
#include <gtest/gtest.h>
#include "NeuralNetwork.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace Ml;

// ---------------------------------------------------------------------------
// Helper: educational banner printed at the start of every test
// ---------------------------------------------------------------------------

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// ======================== Activation Functions ========================

TEST(ActivationFunction, SigmoidBounds) {
    PrintSection("Sigmoid Bounds",
                 "Sigmoid must map 0 -> 0.5, large positive -> ~1, "
                 "large negative -> ~0");

    Sigmoid sig;
    double at_zero   = sig.Activate(0.0);
    double at_pos    = sig.Activate(100.0);
    double at_neg    = sig.Activate(-100.0);

    std::cout << "[Result] Sigmoid(0)    = " << at_zero
              << "  (expected 0.5)" << std::endl;
    std::cout << "[Result] Sigmoid(100)  = " << at_pos
              << "  (expected > 0.99)" << std::endl;
    std::cout << "[Result] Sigmoid(-100) = " << at_neg
              << "  (expected < 0.01)" << std::endl;

    EXPECT_NEAR(at_zero, 0.5, 1e-9);
    EXPECT_GT(at_pos, 0.99);
    EXPECT_LT(at_neg, 0.01);
}

TEST(ActivationFunction, SigmoidDerivativeAtZero) {
    PrintSection("Sigmoid Derivative at Zero",
                 "The derivative at x=0 should be 0.25 "
                 "(maximum slope of the sigmoid curve)");

    Sigmoid sig;
    double d = sig.Derivative(0.0);

    std::cout << "[Result] Sigmoid'(0) = " << d
              << "  (expected 0.25)" << std::endl;

    EXPECT_NEAR(d, 0.25, 1e-9);
}

TEST(ActivationFunction, ReLUPositiveAndNegative) {
    PrintSection("ReLU Positive and Negative",
                 "ReLU passes positive values through and clamps negatives to 0");

    ReLU relu;
    double pos_act  = relu.Activate(5.0);
    double neg_act  = relu.Activate(-3.0);
    double pos_der  = relu.Derivative(5.0);
    double neg_der  = relu.Derivative(-3.0);

    std::cout << "[Result] ReLU(5)    = " << pos_act
              << "  (expected 5.0)" << std::endl;
    std::cout << "[Result] ReLU(-3)   = " << neg_act
              << "  (expected 0.0)" << std::endl;
    std::cout << "[Result] ReLU'(5)   = " << pos_der
              << "  (expected 1.0)" << std::endl;
    std::cout << "[Result] ReLU'(-3)  = " << neg_der
              << "  (expected 0.0)" << std::endl;

    EXPECT_DOUBLE_EQ(pos_act, 5.0);
    EXPECT_DOUBLE_EQ(neg_act, 0.0);
    EXPECT_DOUBLE_EQ(pos_der, 1.0);
    EXPECT_DOUBLE_EQ(neg_der, 0.0);
}

TEST(ActivationFunction, TanhBounds) {
    PrintSection("Tanh Bounds",
                 "tanh(0)=0, tanh(large)~1, tanh(-large)~-1");

    Tanh tanh_fn;
    double at_zero = tanh_fn.Activate(0.0);
    double at_pos  = tanh_fn.Activate(10.0);
    double at_neg  = tanh_fn.Activate(-10.0);

    std::cout << "[Result] tanh(0)   = " << at_zero
              << "  (expected 0.0)" << std::endl;
    std::cout << "[Result] tanh(10)  = " << at_pos
              << "  (expected > 0.99)" << std::endl;
    std::cout << "[Result] tanh(-10) = " << at_neg
              << "  (expected < -0.99)" << std::endl;

    EXPECT_NEAR(at_zero, 0.0, 1e-9);
    EXPECT_GT(at_pos, 0.99);
    EXPECT_LT(at_neg, -0.99);
}

// ======================== Activation Interface Verification ========================

TEST(ActivationInterface, ConcreteClassesInheritBase) {
    PrintSection("Activation Interface Check",
                 "Verify that Sigmoid, ReLU, Tanh inherit from ActivationFunction");

    EXPECT_NE(dynamic_cast<ActivationFunction*>(new Sigmoid()), nullptr);
    EXPECT_NE(dynamic_cast<ActivationFunction*>(new ReLU()), nullptr);
    EXPECT_NE(dynamic_cast<ActivationFunction*>(new Tanh()), nullptr);

    std::cout << "[Result] Sigmoid inherits ActivationFunction: true" << std::endl;
    std::cout << "[Result] ReLU    inherits ActivationFunction: true" << std::endl;
    std::cout << "[Result] Tanh    inherits ActivationFunction: true" << std::endl;
}

// ======================== Layer ========================

TEST(Layer, ForwardProducesCorrectSize) {
    PrintSection("Layer Forward Output Size",
                 "A layer with 3 inputs and 2 neurons must produce exactly "
                 "2 outputs");

    Layer layer(3, 2, std::make_unique<Sigmoid>());
    auto output = layer.Forward({1.0, 2.0, 3.0});

    std::cout << "[Result] output.size() = " << output.size()
              << "  (expected 2)" << std::endl;

    EXPECT_EQ(output.size(), 2u);
}

TEST(Layer, ForwardThrowsOnSizeMismatch) {
    PrintSection("Layer Forward Input Mismatch",
                 "Passing the wrong number of inputs must throw immediately "
                 "rather than produce garbage");

    Layer layer(3, 2, std::make_unique<Sigmoid>());

    std::cout << "[Result] Forward({1.0, 2.0}) on 3-input layer -> "
              << "std::invalid_argument" << std::endl;

    EXPECT_THROW(layer.Forward({1.0, 2.0}), std::invalid_argument);
}

TEST(Layer, ZeroSizeThrows) {
    PrintSection("Layer Zero-Size Construction",
                 "Zero neurons or zero inputs is a configuration error "
                 "that must be caught at construction time");

    std::cout << "[Result] Layer(0,5) -> std::invalid_argument" << std::endl;
    std::cout << "[Result] Layer(5,0) -> std::invalid_argument" << std::endl;

    EXPECT_THROW(Layer(0, 5, std::make_unique<Sigmoid>()), std::invalid_argument);
    EXPECT_THROW(Layer(5, 0, std::make_unique<Sigmoid>()), std::invalid_argument);
}

// ======================== NeuralNetwork construction ========================

TEST(NeuralNetwork, ConstructionValid) {
    PrintSection("Valid Network Construction",
                 "Basic sanity: constructing with valid params must not throw");

    std::cout << "[Result] NeuralNetwork(2, 0.1) -> no throw" << std::endl;

    EXPECT_NO_THROW(NeuralNetwork(2, 0.1));
}

TEST(NeuralNetwork, ConstructionZeroInputThrows) {
    PrintSection("Zero-Input Construction",
                 "A network with 0 inputs is meaningless and must be rejected");

    std::cout << "[Result] NeuralNetwork(0, 0.1) -> std::invalid_argument"
              << std::endl;

    EXPECT_THROW(NeuralNetwork(0, 0.1), std::invalid_argument);
}

TEST(NeuralNetwork, ConstructionBadLearningRateThrows) {
    PrintSection("Invalid Learning Rate",
                 "Learning rate must be in (0, 1]; 0 and >1 are invalid");

    std::cout << "[Result] NeuralNetwork(2, 0.0) -> std::invalid_argument"
              << std::endl;
    std::cout << "[Result] NeuralNetwork(2, 1.5) -> std::invalid_argument"
              << std::endl;

    EXPECT_THROW(NeuralNetwork(2, 0.0), std::invalid_argument);
    EXPECT_THROW(NeuralNetwork(2, 1.5), std::invalid_argument);
}

// ======================== AddLayer ========================

TEST(NeuralNetwork, AddLayerIncreasesCount) {
    PrintSection("AddLayer Count",
                 "Each AddLayer call must increment the layer count by 1");

    NeuralNetwork nn(2);

    std::cout << "[Result] Initial layer count = " << nn.GetLayerCount()
              << "  (expected 0)" << std::endl;
    EXPECT_EQ(nn.GetLayerCount(), 0u);

    nn.AddSigmoidLayer(4);
    std::cout << "[Result] After AddSigmoidLayer(4) = " << nn.GetLayerCount()
              << "  (expected 1)" << std::endl;
    EXPECT_EQ(nn.GetLayerCount(), 1u);

    nn.AddReLULayer(3);
    std::cout << "[Result] After AddReLULayer(3)    = " << nn.GetLayerCount()
              << "  (expected 2)" << std::endl;
    EXPECT_EQ(nn.GetLayerCount(), 2u);
}

TEST(NeuralNetwork, AddLayerZeroNeuronsThrows) {
    PrintSection("AddLayer Zero Neurons",
                 "Adding a layer with 0 neurons is a configuration error");

    NeuralNetwork nn(2);

    std::cout << "[Result] AddSigmoidLayer(0) -> std::invalid_argument"
              << std::endl;

    EXPECT_THROW(nn.AddSigmoidLayer(0), std::invalid_argument);
}

// ======================== Forward ========================

TEST(NeuralNetwork, ForwardNoLayersThrows) {
    PrintSection("Forward With No Layers",
                 "Forwarding through an empty network must throw, "
                 "not silently return the input");

    NeuralNetwork nn(2);

    std::cout << "[Result] Forward on 0-layer network -> std::runtime_error"
              << std::endl;

    EXPECT_THROW(nn.Forward({1.0, 2.0}), std::runtime_error);
}

TEST(NeuralNetwork, ForwardWrongInputSizeThrows) {
    PrintSection("Forward Wrong Input Size",
                 "Input dimension mismatch must be caught at the API level");

    NeuralNetwork nn(2);
    nn.AddSigmoidLayer(3);

    std::cout << "[Result] Forward({1.0}) on 2-input network -> "
              << "std::invalid_argument" << std::endl;

    EXPECT_THROW(nn.Forward({1.0}), std::invalid_argument);
}

TEST(NeuralNetwork, ForwardReturnsCorrectSize) {
    PrintSection("Forward Output Size",
                 "Output vector size must match the last layer's neuron count");

    NeuralNetwork nn(2);
    nn.AddSigmoidLayer(4);
    nn.AddSigmoidLayer(1);
    auto out = nn.Forward({0.5, 0.5});

    std::cout << "[Result] output.size() = " << out.size()
              << "  (expected 1)" << std::endl;

    EXPECT_EQ(out.size(), 1u);
}

TEST(NeuralNetwork, ForwardSigmoidOutputInRange) {
    PrintSection("Sigmoid Output Range",
                 "All-sigmoid network must produce outputs in [0, 1]");

    NeuralNetwork nn(2);
    nn.AddSigmoidLayer(4);
    nn.AddSigmoidLayer(1);
    auto out = nn.Forward({0.5, 0.5});

    std::cout << "[Result] output[0] = " << out[0]
              << "  (expected in [0, 1])" << std::endl;

    EXPECT_GE(out[0], 0.0);
    EXPECT_LE(out[0], 1.0);
}

// ======================== Train ========================

TEST(NeuralNetwork, TrainReducesError) {
    PrintSection("Train Reduces Error",
                 "Repeated training on the same sample must decrease MSE, "
                 "proving backpropagation is working");

    NeuralNetwork nn(2, 0.5);
    nn.AddSigmoidLayer(4);
    nn.AddSigmoidLayer(1);

    std::vector<double> input = {1.0, 0.0};
    std::vector<double> target = {1.0};

    double first_error = nn.Train(input, target);
    double last_error = first_error;
    for (int i = 0; i < 100; ++i) {
        last_error = nn.Train(input, target);
    }

    std::cout << "[Result] first_error = " << std::fixed << std::setprecision(6)
              << first_error << std::endl;
    std::cout << "[Result] last_error  = " << last_error
              << "  (expected < first_error)" << std::endl;

    EXPECT_LT(last_error, first_error);
}

// ======================== TrainBatch ========================

TEST(NeuralNetwork, TrainBatchXORConverges) {
    PrintSection("TrainBatch XOR Convergence",
                 "XOR is the classic non-linearly-separable problem; "
                 "a 2-4-1 sigmoid network should learn it in ~5000 epochs");

    NeuralNetwork nn(2, 0.5);
    nn.AddSigmoidLayer(4);
    nn.AddSigmoidLayer(1);

    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> expected = {
        {0}, {1}, {1}, {0}
    };

    auto errors = nn.TrainBatch(inputs, expected, 5000, 4);

    std::cout << "[Result] initial MSE = " << std::fixed << std::setprecision(6)
              << errors.front() << std::endl;
    std::cout << "[Result] final   MSE = " << errors.back()
              << "  (expected < 0.1)" << std::endl;

    // Show per-input predictions after training
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = nn.Forward(inputs[i]);
        std::cout << "[Result] XOR(" << inputs[i][0] << ", " << inputs[i][1]
                  << ") = " << out[0]
                  << "  (expected " << expected[i][0] << ")" << std::endl;
    }

    EXPECT_LT(errors.back(), errors.front());
    EXPECT_LT(errors.back(), 0.1);
}

TEST(NeuralNetwork, TrainBatchEmptyThrows) {
    PrintSection("TrainBatch Empty Data",
                 "Training on empty data must throw rather than silently "
                 "produce an empty error history");

    NeuralNetwork nn(2);
    nn.AddSigmoidLayer(1);

    std::cout << "[Result] TrainBatch({}, {}, 10) -> std::invalid_argument"
              << std::endl;

    EXPECT_THROW(nn.TrainBatch({}, {}, 10), std::invalid_argument);
}

// ======================== LearningRate ========================

TEST(NeuralNetwork, SetLearningRate) {
    PrintSection("Set Learning Rate",
                 "SetLearningRate must update the value returned by "
                 "GetLearningRate");

    NeuralNetwork nn(2, 0.1);
    nn.SetLearningRate(0.5);
    double rate = nn.GetLearningRate();

    std::cout << "[Result] GetLearningRate() = " << rate
              << "  (expected 0.5)" << std::endl;

    EXPECT_DOUBLE_EQ(rate, 0.5);
}

TEST(NeuralNetwork, SetBadLearningRateThrows) {
    PrintSection("Set Bad Learning Rate",
                 "Out-of-range learning rates must be rejected");

    NeuralNetwork nn(2, 0.1);

    std::cout << "[Result] SetLearningRate(0.0) -> std::invalid_argument"
              << std::endl;
    std::cout << "[Result] SetLearningRate(2.0) -> std::invalid_argument"
              << std::endl;

    EXPECT_THROW(nn.SetLearningRate(0.0), std::invalid_argument);
    EXPECT_THROW(nn.SetLearningRate(2.0), std::invalid_argument);
}
