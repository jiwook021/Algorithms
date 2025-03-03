/**
 * @file main.cpp
 * @brief Driver for the NeuralNetwork module -- XOR demo.
 */
#include "NeuralNetwork.hpp"
#include <iostream>
#include <iomanip>

int main() {
    try {
        std::cout << "=== XOR Problem Demo ===\n";

        std::vector<std::vector<double>> inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<std::vector<double>> expected = {
            {0}, {1}, {1}, {0}
        };

        Ml::NeuralNetwork nn(2, 0.5);
        nn.AddSigmoidLayer(4);
        nn.AddSigmoidLayer(1);
        nn.PrintArchitecture();

        std::cout << "\nTraining (10 000 epochs)...\n";
        auto errors = nn.TrainBatch(inputs, expected, 10000, 4);

        std::cout << "Final MSE: " << std::fixed << std::setprecision(6)
                  << errors.back() << "\n\n";

        std::cout << "Results:\n";
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto result = nn.Forward(inputs[i]);
            std::cout << "  [" << inputs[i][0] << ", " << inputs[i][1]
                      << "] -> " << std::fixed << std::setprecision(4)
                      << result[0]
                      << "  (expected " << expected[i][0] << ")\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
