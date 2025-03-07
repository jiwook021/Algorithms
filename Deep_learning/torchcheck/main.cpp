#include <torch/torch.h>
#include <iostream>

int main() {
    try {
        // Create a tensor with random values
        torch::Tensor tensor = torch::rand({3, 3});

        // Perform a simple tensor operation
        torch::Tensor result = tensor * tensor;

        // Output the tensor
        std::cout << "Tensor:\n" << tensor << "\n";
        std::cout << "Result (tensor squared):\n" << result << "\n";

        std::cout << "Torch is working correctly!\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
