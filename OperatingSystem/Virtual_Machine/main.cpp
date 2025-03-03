/**
 * @file main.cpp
 * @brief Driver for VirtualMachine
 */

#include "VirtualMachine.hpp"
#include <iostream>

int main() {
    try {
        VirtualMachine Vm;
        
        // Test 1: Factorial program
        std::cout << "\n========== Test 1: Factorial of 5 ==========\n";
        auto FactorialProgram = TestPrograms::CreateFactorialProgram();
        Vm.LoadProgram(FactorialProgram);
        Vm.Run();
        // Expected output: 120
        
        // Test 2: Fibonacci sequence
        std::cout << "\n========== Test 2: Fibonacci Sequence ==========\n";
        auto FibonacciProgram = TestPrograms::CreateFibonacciProgram();
        Vm.LoadProgram(FibonacciProgram);
        Vm.Run();
        // Expected output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        
        // Test 3: Heap memory operations
        std::cout << "\n========== Test 3: Heap Memory Operations ==========\n";
        auto HeapTestProgram = TestPrograms::CreateHeapTestProgram();
        Vm.LoadProgram(HeapTestProgram);
        Vm.Run();
        // Expected output: Values 1 through 10
        
        std::cout << "\n========== All tests completed successfully ==========\n";
        
    } catch (const VMException& e) {
        std::cerr << "VM Exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}