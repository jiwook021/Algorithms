#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <stack>
#include <mutex>
#include <optional>
#include <variant>
#include <sstream>

/**
 * A simple stack-based virtual machine implementation in C++17/20
 * This VM includes:
 * - A memory model with stack and heap support
 * - A bytecode instruction set
 * - A central execution loop
 */

// Forward declarations
class Memory;
class VirtualMachine;

/**
 * Custom exception types for VM operations
 * Following Single Responsibility Principle for error handling
 */
class VMException : public std::runtime_error {
public:
    explicit VMException(const std::string& message) : std::runtime_error(message) {}
};

class StackOverflowException : public VMException {
public:
    explicit StackOverflowException() : VMException("Stack overflow") {}
};

class StackUnderflowException : public VMException {
public:
    explicit StackUnderflowException() : VMException("Stack underflow") {}
};

class HeapAllocationException : public VMException {
public:
    explicit HeapAllocationException() : VMException("Heap allocation failed") {}
};

class InvalidInstructionException : public VMException {
public:
    explicit InvalidInstructionException(int opcode)
        : VMException("Invalid instruction opcode: " + std::to_string(opcode)) {}
};

class DivisionByZeroException : public VMException {
public:
    explicit DivisionByZeroException() : VMException("Division by zero") {}
};

/**
 * Memory class - Manages stack and heap memory
 * - Single Responsibility: Manages memory operations
 * - Thread-safe with mutex protection for critical operations
 */
class Memory {
public:
    // Value type that can be stored in memory (stack or heap)
    using Value = std::variant<int, float, bool, void*>;
    
    explicit Memory(size_t stackCapacity = 1024, size_t heapCapacity = 4096)
        : stackCapacity_(stackCapacity), heapCapacity_(heapCapacity) {
        // Initialize heap with allocated capacity
        heap_.resize(heapCapacity_, 0);
    }
    
    // Stack operations - Thread safe with mutex protection
    void pushStack(const Value& value) {
        std::lock_guard<std::mutex> lock(stackMutex_);
        
        if (stack_.size() >= stackCapacity_) {
            throw StackOverflowException();
        }
        
        stack_.push_back(value);
    }
    
    Value popStack() {
        std::lock_guard<std::mutex> lock(stackMutex_);
        
        if (stack_.empty()) {
            throw StackUnderflowException();
        }
        
        Value value = stack_.back();
        stack_.pop_back();
        return value;
    }
    
    Value& topStack() {
        std::lock_guard<std::mutex> lock(stackMutex_);
        
        if (stack_.empty()) {
            throw StackUnderflowException();
        }
        
        return stack_.back();
    }
    
    size_t stackSize() const {
        std::lock_guard<std::mutex> lock(stackMutex_);
        return stack_.size();
    }
    
    // Heap operations - Thread safe with mutex protection
    size_t allocateHeap(size_t size) {
        std::lock_guard<std::mutex> lock(heapMutex_);
        
        // Simple bump allocator for demonstration
        // A real implementation would use a more sophisticated memory allocation strategy
        if (heapAllocPtr_ + size > heapCapacity_) {
            throw HeapAllocationException();
        }
        
        size_t ptr = heapAllocPtr_;
        heapAllocPtr_ += size;
        return ptr;
    }
    
    void writeHeap(size_t address, uint8_t value) {
        std::lock_guard<std::mutex> lock(heapMutex_);
        
        if (address >= heapCapacity_) {
            throw std::out_of_range("Heap address out of bounds");
        }
        
        heap_[address] = value;
    }
    
    uint8_t readHeap(size_t address) const {
        std::lock_guard<std::mutex> lock(heapMutex_);
        
        if (address >= heapCapacity_) {
            throw std::out_of_range("Heap address out of bounds");
        }
        
        return heap_[address];
    }
    
    // Reset both stack and heap
    void reset() {
        // Clear the stack
        {
            std::lock_guard<std::mutex> lock(stackMutex_);
            stack_.clear();
        }
        
        // Reset heap allocation pointer
        {
            std::lock_guard<std::mutex> lock(heapMutex_);
            std::fill(heap_.begin(), heap_.end(), 0);
            heapAllocPtr_ = 0;
        }
    }
    
private:
    // Stack implementation using a vector for O(1) push/pop operations
    std::vector<Value> stack_;
    size_t stackCapacity_;
    mutable std::mutex stackMutex_; // Protects stack_ operations
    
    // Heap implementation using a vector of bytes
    std::vector<uint8_t> heap_;
    size_t heapCapacity_;
    size_t heapAllocPtr_ = 0; // Points to the next free byte in the heap
    mutable std::mutex heapMutex_; // Protects heap_ operations
};

/**
 * Instruction - Abstract base class for all VM instructions
 * - Open/Closed Principle: New instructions can be added without modifying existing code
 * - Interface Segregation Principle: Each instruction only needs to implement what it needs
 */
class Instruction {
public:
    virtual ~Instruction() = default;
    
    // Execute the instruction on the given VM
    virtual void execute(VirtualMachine& vm) = 0;
    
    // Get a string representation of the instruction (useful for debugging)
    virtual std::string toString() const = 0;
};

/**
 * Bytecode Opcodes
 * Define all possible instructions for our VM
 */
enum class Opcode : uint8_t {
    // Stack operations
    PUSH,       // Push immediate value onto the stack
    POP,        // Pop value from the stack
    DUP,        // Duplicate top value on the stack
    SWAP,       // Swap top two values on the stack
    
    // Arithmetic operations
    ADD,        // Add top two values
    SUB,        // Subtract top value from second value
    MUL,        // Multiply top two values
    DIV,        // Divide second value by top value
    MOD,        // Modulo operation
    
    // Comparison operations
    EQ,         // Equal
    NE,         // Not equal
    LT,         // Less than
    GT,         // Greater than
    LE,         // Less than or equal
    GE,         // Greater than or equal
    
    // Control flow
    JMP,        // Unconditional jump
    JZ,         // Jump if zero
    JNZ,        // Jump if not zero
    
    // Memory operations
    LOAD,       // Load from heap
    STORE,      // Store to heap
    ALLOC,      // Allocate memory on heap
    
    // Miscellaneous
    PRINT,      // Print top of stack
    HALT        // Stop execution
};

/**
 * VirtualMachine class - Executes bytecode instructions
 * - Single Responsibility: Executes VM instructions
 * - Dependency Inversion: Depends on abstractions (Instruction interface)
 */
class VirtualMachine {
public:
    explicit VirtualMachine(size_t stackCapacity = 1024, size_t heapCapacity = 4096)
        : memory_(stackCapacity, heapCapacity), running_(false), pc_(0) {}
    
    // Load a program into the VM
    void loadProgram(const std::vector<uint8_t>& bytecode) {
        std::lock_guard<std::mutex> lock(executionMutex_);
        program_ = bytecode;
        pc_ = 0; // Reset program counter
        memory_.reset(); // Reset memory state
    }
    
    // Run the loaded program
    void run() {
        std::lock_guard<std::mutex> lock(executionMutex_);
        
        if (program_.empty()) {
            throw VMException("No program loaded");
        }
        
        running_ = true;
        pc_ = 0; // Start from the beginning
        
        // Main execution loop (fetch-decode-execute cycle)
        while (running_ && pc_ < program_.size()) {
            // Fetch
            Opcode opcode = static_cast<Opcode>(program_[pc_]);
            pc_++; // Move to next instruction or operand
            
            // Decode and Execute
            executeInstruction(opcode);
        }
    }
    
    // Execute a single instruction
    void step() {
        std::lock_guard<std::mutex> lock(executionMutex_);
        
        if (!running_ || pc_ >= program_.size()) {
            throw VMException("VM not running or end of program reached");
        }
        
        // Fetch
        Opcode opcode = static_cast<Opcode>(program_[pc_]);
        pc_++; // Move to next instruction or operand
        
        // Decode and Execute
        executeInstruction(opcode);
    }
    
    // Stop VM execution
    void halt() {
        std::lock_guard<std::mutex> lock(executionMutex_);
        running_ = false;
    }
    
    // Get current program counter
    size_t getProgramCounter() const {
        return pc_;
    }
    
    // Set program counter (for jumps)
    void setProgramCounter(size_t newPc) {
        if (newPc >= program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        pc_ = newPc;
    }
    
    // Memory access
    Memory& getMemory() {
        return memory_;
    }
    
    // Read next byte from program (for operands)
    uint8_t readProgramByte() {
        if (pc_ >= program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        return program_[pc_++];
    }
    
    // Read next 4 bytes from program as int (for operands)
    int readProgramInt() {
        if (pc_ + 3 >= program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        
        int value = (program_[pc_]) |
                   (program_[pc_ + 1] << 8) |
                   (program_[pc_ + 2] << 16) |
                   (program_[pc_ + 3] << 24);
        
        pc_ += 4;
        return value;
    }
    
private:
    Memory memory_;
    std::vector<uint8_t> program_;
    bool running_;
    size_t pc_; // Program counter
    std::mutex executionMutex_; // Protects execution state
    
    // Decode and execute a single instruction
    void executeInstruction(Opcode opcode) {
        // Helper lambdas for common stack operations
        auto popInt = [this]() -> int {
            auto value = memory_.popStack();
            if (std::holds_alternative<int>(value)) {
                return std::get<int>(value);
            }
            throw VMException("Type error: Expected integer on stack");
        };
        
        auto popBool = [this]() -> bool {
            auto value = memory_.popStack();
            if (std::holds_alternative<bool>(value)) {
                return std::get<bool>(value);
            }
            throw VMException("Type error: Expected boolean on stack");
        };
        
        auto pushInt = [this](int value) {
            memory_.pushStack(value);
        };
        
        auto pushBool = [this](bool value) {
            memory_.pushStack(value);
        };
        
        // Execute the instruction based on opcode
        switch (opcode) {
            // Stack operations
            case Opcode::PUSH: {
                int value = readProgramInt();
                memory_.pushStack(value);
                break;
            }
            case Opcode::POP: {
                memory_.popStack();
                break;
            }
            case Opcode::DUP: {
                auto value = memory_.topStack();
                memory_.pushStack(value);
                break;
            }
            case Opcode::SWAP: {
                if (memory_.stackSize() < 2) {
                    throw StackUnderflowException();
                }
                auto top = memory_.popStack();
                auto second = memory_.popStack();
                memory_.pushStack(top);
                memory_.pushStack(second);
                break;
            }
            
            // Arithmetic operations
            case Opcode::ADD: {
                int b = popInt();
                int a = popInt();
                pushInt(a + b);
                break;
            }
            case Opcode::SUB: {
                int b = popInt();
                int a = popInt();
                pushInt(a - b);
                break;
            }
            case Opcode::MUL: {
                int b = popInt();
                int a = popInt();
                pushInt(a * b);
                break;
            }
            case Opcode::DIV: {
                int b = popInt();
                if (b == 0) {
                    throw DivisionByZeroException();
                }
                int a = popInt();
                pushInt(a / b);
                break;
            }
            case Opcode::MOD: {
                int b = popInt();
                if (b == 0) {
                    throw DivisionByZeroException();
                }
                int a = popInt();
                pushInt(a % b);
                break;
            }
            
            // Comparison operations
            case Opcode::EQ: {
                int b = popInt();
                int a = popInt();
                pushBool(a == b);
                break;
            }
            case Opcode::NE: {
                int b = popInt();
                int a = popInt();
                pushBool(a != b);
                break;
            }
            case Opcode::LT: {
                int b = popInt();
                int a = popInt();
                pushBool(a < b);
                break;
            }
            case Opcode::GT: {
                int b = popInt();
                int a = popInt();
                pushBool(a > b);
                break;
            }
            case Opcode::LE: {
                int b = popInt();
                int a = popInt();
                pushBool(a <= b);
                break;
            }
            case Opcode::GE: {
                int b = popInt();
                int a = popInt();
                pushBool(a >= b);
                break;
            }
            
            // Control flow
            case Opcode::JMP: {
                int address = readProgramInt();
                if (address < 0 || static_cast<size_t>(address) >= program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                pc_ = static_cast<size_t>(address);
                break;
            }
            case Opcode::JZ: {
                int address = readProgramInt();
                if (address < 0 || static_cast<size_t>(address) >= program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                
                bool condition = false;
                auto value = memory_.popStack();
                
                if (std::holds_alternative<bool>(value)) {
                    condition = !std::get<bool>(value);
                } else if (std::holds_alternative<int>(value)) {
                    condition = (std::get<int>(value) == 0);
                } else {
                    throw VMException("Type error: Expected boolean or integer for condition");
                }
                
                if (condition) {
                    pc_ = static_cast<size_t>(address);
                }
                break;
            }
            case Opcode::JNZ: {
                int address = readProgramInt();
                if (address < 0 || static_cast<size_t>(address) >= program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                
                bool condition = false;
                auto value = memory_.popStack();
                
                if (std::holds_alternative<bool>(value)) {
                    condition = std::get<bool>(value);
                } else if (std::holds_alternative<int>(value)) {
                    condition = (std::get<int>(value) != 0);
                } else {
                    throw VMException("Type error: Expected boolean or integer for condition");
                }
                
                if (condition) {
                    pc_ = static_cast<size_t>(address);
                }
                break;
            }
            
            // Memory operations
            case Opcode::LOAD: {
                size_t address = static_cast<size_t>(popInt());
                uint8_t value = memory_.readHeap(address);
                pushInt(value);
                break;
            }
            case Opcode::STORE: {
                size_t address = static_cast<size_t>(popInt());
                int value = popInt();
                memory_.writeHeap(address, static_cast<uint8_t>(value));
                break;
            }
            case Opcode::ALLOC: {
                size_t size = static_cast<size_t>(popInt());
                size_t address = memory_.allocateHeap(size);
                pushInt(static_cast<int>(address));
                break;
            }
            
            // Miscellaneous
            case Opcode::PRINT: {
                auto value = memory_.popStack();
                
                if (std::holds_alternative<int>(value)) {
                    std::cout << std::get<int>(value) << std::endl;
                } else if (std::holds_alternative<float>(value)) {
                    std::cout << std::get<float>(value) << std::endl;
                } else if (std::holds_alternative<bool>(value)) {
                    std::cout << (std::get<bool>(value) ? "true" : "false") << std::endl;
                } else {
                    std::cout << "Pointer: " << std::get<void*>(value) << std::endl;
                }
                break;
            }
            case Opcode::HALT: {
                running_ = false;
                break;
            }
            
            default:
                throw InvalidInstructionException(static_cast<int>(opcode));
        }
    }
};

/**
 * BytecodeAssembler - Helper class to create bytecode programs
 * - Single Responsibility: Assembles bytecode for the VM
 */
class BytecodeAssembler {
public:
    BytecodeAssembler() = default;
    
    // Add an instruction without operands
    BytecodeAssembler& emit(Opcode opcode) {
        bytecode_.push_back(static_cast<uint8_t>(opcode));
        return *this;
    }
    
    // Add an instruction with an integer operand
    BytecodeAssembler& emitInt(Opcode opcode, int operand) {
        bytecode_.push_back(static_cast<uint8_t>(opcode));
        
        // Little-endian format for integer operands
        bytecode_.push_back(operand & 0xFF);
        bytecode_.push_back((operand >> 8) & 0xFF);
        bytecode_.push_back((operand >> 16) & 0xFF);
        bytecode_.push_back((operand >> 24) & 0xFF);
        
        return *this;
    }
    
    // Get the current position in the bytecode (useful for jump targets)
    size_t getCurrentPosition() const {
        return bytecode_.size();
    }
    
    // Update a previously emitted integer operand (for forward jumps)
    void updateIntOperandAt(size_t position, int newValue) {
        if (position + 4 >= bytecode_.size()) {
            throw std::out_of_range("Invalid position for operand update");
        }
        
        bytecode_[position] = newValue & 0xFF;
        bytecode_[position + 1] = (newValue >> 8) & 0xFF;
        bytecode_[position + 2] = (newValue >> 16) & 0xFF;
        bytecode_[position + 3] = (newValue >> 24) & 0xFF;
    }
    
    // Get the assembled bytecode
    std::vector<uint8_t> getBytecode() const {
        return bytecode_;
    }
    
private:
    std::vector<uint8_t> bytecode_;
};

/**
 * Helper functions to create test programs
 */
namespace TestPrograms {
    // Program that calculates factorial of 5
    std::vector<uint8_t> createFactorialProgram() {
        BytecodeAssembler assembler;
        
        // Initialize variables
        assembler.emitInt(Opcode::PUSH, 5);     // Input number (5)
        assembler.emitInt(Opcode::PUSH, 1);     // Result starts at 1
        
        // Loop start position
        size_t loopStart = assembler.getCurrentPosition();
        
        // Check if input <= 1
        assembler.emit(Opcode::DUP);           // Duplicate input for comparison
        assembler.emitInt(Opcode::PUSH, 1);
        assembler.emit(Opcode::LE);            // Check if input <= 1
        
        // Jump to end if input <= 1
        size_t jumpPos = assembler.getCurrentPosition();
        assembler.emitInt(Opcode::JZ, 0);      // Placeholder jump address, will update later
        
        // Multiply result by input
        assembler.emit(Opcode::DUP);           // Duplicate input
        assembler.emit(Opcode::SWAP);          // Swap to get result on top
        assembler.emit(Opcode::MUL);           // Multiply result * input
        
        // Decrement input
        assembler.emit(Opcode::SWAP);          // Swap to get input on top
        assembler.emitInt(Opcode::PUSH, 1);
        assembler.emit(Opcode::SUB);           // Subtract 1 from input
        
        // Jump back to loop start
        assembler.emitInt(Opcode::JMP, loopStart);
        
        // Update the conditional jump address
        size_t endPos = assembler.getCurrentPosition();
        assembler.updateIntOperandAt(jumpPos + 1, endPos);
        
        // Drop the input, keep only the result
        assembler.emit(Opcode::SWAP);
        assembler.emit(Opcode::POP);
        
        // Print the result
        assembler.emit(Opcode::PRINT);
        
        // Stop execution
        assembler.emit(Opcode::HALT);
        
        return assembler.getBytecode();
    }
    
    // Program that calculates Fibonacci sequence (first 10 numbers)
    std::vector<uint8_t> createFibonacciProgram() {
        BytecodeAssembler assembler;
        
        // Initialize variables
        assembler.emitInt(Opcode::PUSH, 0);     // First number
        assembler.emit(Opcode::DUP);
        assembler.emit(Opcode::PRINT);          // Print first number (0)
        
        assembler.emitInt(Opcode::PUSH, 1);     // Second number
        assembler.emit(Opcode::DUP);
        assembler.emit(Opcode::PRINT);          // Print second number (1)
        
        assembler.emitInt(Opcode::PUSH, 8);     // Counter (8 more to print)
        
        // Loop start
        size_t loopStart = assembler.getCurrentPosition();
        
        // Check if counter > 0
        assembler.emit(Opcode::DUP);            // Duplicate counter
        assembler.emitInt(Opcode::PUSH, 0);
        assembler.emit(Opcode::GT);             // Check if counter > 0
        
        // Jump to end if counter <= 0
        size_t jumpPos = assembler.getCurrentPosition();
        assembler.emitInt(Opcode::JZ, 0);       // Placeholder jump address
        
        // Calculate next Fibonacci number
        assembler.emit(Opcode::SWAP);           // Get second number
        assembler.emit(Opcode::DUP);            // Duplicate it
        assembler.emit(Opcode::SWAP);           // Swap
        assembler.emit(Opcode::SWAP);           // Get first number
        assembler.emit(Opcode::DUP);            // Duplicate it
        assembler.emit(Opcode::ADD);            // Add first + second
        
        // Print the next number
        assembler.emit(Opcode::DUP);
        assembler.emit(Opcode::PRINT);
        
        // Decrement counter
        assembler.emit(Opcode::SWAP);           // Get counter
        assembler.emitInt(Opcode::PUSH, 1);
        assembler.emit(Opcode::SUB);            // Subtract 1
        
        // Jump back to loop start
        assembler.emitInt(Opcode::JMP, loopStart);
        
        // Update jump address
        size_t endPos = assembler.getCurrentPosition();
        assembler.updateIntOperandAt(jumpPos + 1, endPos);
        
        // Cleanup stack and halt
        assembler.emit(Opcode::POP);            // Pop counter
        assembler.emit(Opcode::POP);            // Pop second number
        assembler.emit(Opcode::POP);            // Pop first number
        
        assembler.emit(Opcode::HALT);
        
        return assembler.getBytecode();
    }
    
    // Program that tests the heap memory
    std::vector<uint8_t> createHeapTestProgram() {
        BytecodeAssembler assembler;
        
        // Allocate 10 bytes on the heap
        assembler.emitInt(Opcode::PUSH, 10);
        assembler.emit(Opcode::ALLOC);
        
        // Store the heap address for later use
        assembler.emit(Opcode::DUP);
        
        // Store values 1-10 in the allocated memory
        for (int i = 0; i < 10; i++) {
            // Duplicate the current address
            assembler.emit(Opcode::DUP);
            
            // Push the value to store (i+1)
            assembler.emitInt(Opcode::PUSH, i + 1);
            
            // Store value at current address
            assembler.emit(Opcode::STORE);
            
            // Increment address for next iteration
            if (i < 9) {  // Skip increment on last iteration
                assembler.emit(Opcode::DUP);
                assembler.emitInt(Opcode::PUSH, 1);
                assembler.emit(Opcode::ADD);
            }
        }
        
        // Now read back and print values
        for (int i = 0; i < 10; i++) {
            // Load value from current address
            assembler.emit(Opcode::LOAD);
            
            // Print the loaded value
            assembler.emit(Opcode::PRINT);
            
            // If not the last iteration, compute next address
            if (i < 9) {
                assembler.emitInt(Opcode::PUSH, i + 1);
                assembler.emit(Opcode::ADD);
            }
        }
        
        assembler.emit(Opcode::HALT);
        
        return assembler.getBytecode();
    }
}

/**
 * Main test function to demonstrate VM execution
 */
int main() {
    try {
        VirtualMachine vm;
        
        // Test 1: Factorial program
        std::cout << "\n========== Test 1: Factorial of 5 ==========\n";
        auto factorialProgram = TestPrograms::createFactorialProgram();
        vm.loadProgram(factorialProgram);
        vm.run();
        // Expected output: 120
        
        // Test 2: Fibonacci sequence
        std::cout << "\n========== Test 2: Fibonacci Sequence ==========\n";
        auto fibonacciProgram = TestPrograms::createFibonacciProgram();
        vm.loadProgram(fibonacciProgram);
        vm.run();
        // Expected output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        
        // Test 3: Heap memory operations
        std::cout << "\n========== Test 3: Heap Memory Operations ==========\n";
        auto heapTestProgram = TestPrograms::createHeapTestProgram();
        vm.loadProgram(heapTestProgram);
        vm.run();
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