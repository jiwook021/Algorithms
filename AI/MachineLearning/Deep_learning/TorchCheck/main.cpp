#include <Torch/Torch.h>
#include <iostream>

int main() {
    try {
        // Create a tensor with random values
        Torch::Tensor Tensor = Torch::rand({3, 3});

        // Perform a simple tensor operation
        Torch::Tensor Result = Tensor * Tensor;

        // Output the tensor
        std::cout << "Tensor:\n" << Tensor << "\n";
        std::cout << "Result (tensor squared):\n" << Result << "\n";

        std::cout << "Torch is working correctly!\n";
    }
    catch (const C10::Error& e) {
        std::cerr << "Torch error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
