/**
 * @file VirtualMachine.hpp
 * @brief Stack-based bytecode virtual machine interpreter
 * @details Implements Memory, Instruction set, VirtualMachine stack machine, and BytecodeAssembler. Executes a simple bytecode instruction set with arithmetic, comparison, jump, and call/return instructions.
 */

#pragma once

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
    explicit VMException(const std::string& Message) : std::runtime_error(Message) {}
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
    explicit InvalidInstructionException(int Opcode)
        : VMException("Invalid instruction opcode: " + std::to_string(Opcode)) {}
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
    
    explicit Memory(size_t StackCapacity = 1024, size_t HeapCapacity = 4096)
        : StackCapacity_(StackCapacity), HeapCapacity_(HeapCapacity) {
        // Initialize heap with allocated capacity
        Heap_.resize(HeapCapacity_, 0);
    }
    
    // Stack operations - Thread safe with mutex protection
    void PushStack(const Value& value) {
        std::lock_guard<std::mutex> Lock(StackMutex_);
        
        if (stack_.size() >= StackCapacity_) {
            throw StackOverflowException();
        }
        
        stack_.push_back(value);
    }
    
    Value PopStack() {
        std::lock_guard<std::mutex> Lock(StackMutex_);
        
        if (stack_.empty()) {
            throw StackUnderflowException();
        }
        
        Value value = stack_.back();
        stack_.pop_back();
        return value;
    }
    
    Value& TopStack() {
        std::lock_guard<std::mutex> Lock(StackMutex_);
        
        if (stack_.empty()) {
            throw StackUnderflowException();
        }
        
        return stack_.back();
    }
    
    size_t StackSize() const {
        std::lock_guard<std::mutex> Lock(StackMutex_);
        return stack_.size();
    }
    
    // Heap operations - Thread safe with mutex protection
    size_t AllocateHeap(size_t size) {
        std::lock_guard<std::mutex> Lock(HeapMutex_);
        
        // Simple bump allocator for demonstration
        // A real implementation would use a more sophisticated memory allocation strategy
        if (HeapAllocPtr_ + size > HeapCapacity_) {
            throw HeapAllocationException();
        }
        
        size_t Ptr = HeapAllocPtr_;
        HeapAllocPtr_ += size;
        return Ptr;
    }
    
    void WriteHeap(size_t Address, uint8_t value) {
        std::lock_guard<std::mutex> Lock(HeapMutex_);
        
        if (Address >= HeapCapacity_) {
            throw std::out_of_range("Heap address out of bounds");
        }
        
        Heap_[Address] = value;
    }
    
    uint8_t ReadHeap(size_t Address) const {
        std::lock_guard<std::mutex> Lock(HeapMutex_);
        
        if (Address >= HeapCapacity_) {
            throw std::out_of_range("Heap address out of bounds");
        }
        
        return Heap_[Address];
    }
    
    // Reset both stack and heap
    void Reset() {
        // Clear the stack
        {
            std::lock_guard<std::mutex> Lock(StackMutex_);
            stack_.clear();
        }
        
        // Reset heap allocation pointer
        {
            std::lock_guard<std::mutex> Lock(HeapMutex_);
            std::fill(Heap_.begin(), Heap_.end(), 0);
            HeapAllocPtr_ = 0;
        }
    }
    
private:
    // Stack implementation using a vector for O(1) push/pop operations
    std::vector<Value> stack_;
    size_t StackCapacity_;
    mutable std::mutex StackMutex_; // Protects stack_ operations
    
    // Heap implementation using a vector of bytes
    std::vector<uint8_t> Heap_;
    size_t HeapCapacity_;
    size_t HeapAllocPtr_ = 0; // Points to the next free byte in the heap
    mutable std::mutex HeapMutex_; // Protects heap_ operations
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
    virtual void Execute(VirtualMachine& Vm) = 0;
    
    // Get a string representation of the instruction (useful for debugging)
    virtual std::string ToString() const = 0;
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
    explicit VirtualMachine(size_t StackCapacity = 1024, size_t HeapCapacity = 4096)
        : Memory_(StackCapacity, HeapCapacity), Running_(false), Pc_(0) {}
    
    // Load a program into the VM
    void LoadProgram(const std::vector<uint8_t>& Bytecode) {
        std::lock_guard<std::mutex> Lock(ExecutionMutex_);
        Program_ = Bytecode;
        Pc_ = 0; // Reset program counter
        Memory_.Reset(); // Reset memory state
    }
    
    // Run the loaded program
    void Run() {
        std::lock_guard<std::mutex> Lock(ExecutionMutex_);
        
        if (Program_.empty()) {
            throw VMException("No program loaded");
        }
        
        Running_ = true;
        Pc_ = 0; // Start from the beginning
        
        // Main execution loop (fetch-decode-execute cycle)
        while (Running_ && Pc_ < Program_.size()) {
            // Fetch
            Opcode Op = static_cast<Opcode>(Program_[Pc_]);
            Pc_++; // Move to next instruction or operand
            
            // Decode and Execute
            ExecuteInstruction(Op);
        }
    }
    
    // Execute a single instruction
    void Step() {
        std::lock_guard<std::mutex> Lock(ExecutionMutex_);
        
        if (!Running_ || Pc_ >= Program_.size()) {
            throw VMException("VM not running or end of program reached");
        }
        
        // Fetch
        Opcode Op = static_cast<Opcode>(Program_[Pc_]);
        Pc_++; // Move to next instruction or operand
        
        // Decode and Execute
        ExecuteInstruction(Op);
    }
    
    // Stop VM execution
    void Halt() {
        std::lock_guard<std::mutex> Lock(ExecutionMutex_);
        Running_ = false;
    }
    
    // Get current program counter
    size_t GetProgramCounter() const {
        return Pc_;
    }
    
    // Set program counter (for jumps)
    void SetProgramCounter(size_t NewPc) {
        if (NewPc >= Program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        Pc_ = NewPc;
    }
    
    // Memory access
    Memory& GetMemory() {
        return Memory_;
    }
    
    // Read next byte from program (for operands)
    uint8_t ReadProgramByte() {
        if (Pc_ >= Program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        return Program_[Pc_++];
    }
    
    // Read next 4 bytes from program as int (for operands)
    int ReadProgramInt() {
        if (Pc_ + 3 >= Program_.size()) {
            throw std::out_of_range("Program counter out of bounds");
        }
        
        int value = (Program_[Pc_]) |
                   (Program_[Pc_ + 1] << 8) |
                   (Program_[Pc_ + 2] << 16) |
                   (Program_[Pc_ + 3] << 24);
        
        Pc_ += 4;
        return value;
    }
    
private:
    Memory Memory_;
    std::vector<uint8_t> Program_;
    bool Running_;
    size_t Pc_; // Program counter
    std::mutex ExecutionMutex_; // Protects execution state
    
    // Decode and execute a single instruction
    void ExecuteInstruction(Opcode Op) {
        // Helper lambdas for common stack operations
        auto PopInt = [this]() -> int {
            auto value = Memory_.PopStack();
            if (std::holds_alternative<int>(value)) {
                return std::get<int>(value);
            }
            throw VMException("Type error: Expected integer on stack");
        };
        
        auto PopBool = [this]() -> bool {
            auto value = Memory_.PopStack();
            if (std::holds_alternative<bool>(value)) {
                return std::get<bool>(value);
            }
            throw VMException("Type error: Expected boolean on stack");
        };
        
        auto PushInt = [this](int value) {
            Memory_.PushStack(value);
        };
        
        auto PushBool = [this](bool value) {
            Memory_.PushStack(value);
        };
        
        // Execute the instruction based on opcode
        switch (Op) {
            // Stack operations
            case Opcode::PUSH: {
                int value = ReadProgramInt();
                Memory_.PushStack(value);
                break;
            }
            case Opcode::POP: {
                Memory_.PopStack();
                break;
            }
            case Opcode::DUP: {
                auto value = Memory_.TopStack();
                Memory_.PushStack(value);
                break;
            }
            case Opcode::SWAP: {
                if (Memory_.StackSize() < 2) {
                    throw StackUnderflowException();
                }
                auto top = Memory_.PopStack();
                auto second = Memory_.PopStack();
                Memory_.PushStack(top);
                Memory_.PushStack(second);
                break;
            }
            
            // Arithmetic operations
            case Opcode::ADD: {
                int b = PopInt();
                int a = PopInt();
                PushInt(a + b);
                break;
            }
            case Opcode::SUB: {
                int b = PopInt();
                int a = PopInt();
                PushInt(a - b);
                break;
            }
            case Opcode::MUL: {
                int b = PopInt();
                int a = PopInt();
                PushInt(a * b);
                break;
            }
            case Opcode::DIV: {
                int b = PopInt();
                if (b == 0) {
                    throw DivisionByZeroException();
                }
                int a = PopInt();
                PushInt(a / b);
                break;
            }
            case Opcode::MOD: {
                int b = PopInt();
                if (b == 0) {
                    throw DivisionByZeroException();
                }
                int a = PopInt();
                PushInt(a % b);
                break;
            }
            
            // Comparison operations
            case Opcode::EQ: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a == b);
                break;
            }
            case Opcode::NE: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a != b);
                break;
            }
            case Opcode::LT: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a < b);
                break;
            }
            case Opcode::GT: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a > b);
                break;
            }
            case Opcode::LE: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a <= b);
                break;
            }
            case Opcode::GE: {
                int b = PopInt();
                int a = PopInt();
                PushBool(a >= b);
                break;
            }
            
            // Control flow
            case Opcode::JMP: {
                int Address = ReadProgramInt();
                if (Address < 0 || static_cast<size_t>(Address) >= Program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                Pc_ = static_cast<size_t>(Address);
                break;
            }
            case Opcode::JZ: {
                int Address = ReadProgramInt();
                if (Address < 0 || static_cast<size_t>(Address) >= Program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                
                bool Condition = false;
                auto value = Memory_.PopStack();
                
                if (std::holds_alternative<bool>(value)) {
                    Condition = !std::get<bool>(value);
                } else if (std::holds_alternative<int>(value)) {
                    Condition = (std::get<int>(value) == 0);
                } else {
                    throw VMException("Type error: Expected boolean or integer for condition");
                }
                
                if (Condition) {
                    Pc_ = static_cast<size_t>(Address);
                }
                break;
            }
            case Opcode::JNZ: {
                int Address = ReadProgramInt();
                if (Address < 0 || static_cast<size_t>(Address) >= Program_.size()) {
                    throw std::out_of_range("Jump address out of bounds");
                }
                
                bool Condition = false;
                auto value = Memory_.PopStack();
                
                if (std::holds_alternative<bool>(value)) {
                    Condition = std::get<bool>(value);
                } else if (std::holds_alternative<int>(value)) {
                    Condition = (std::get<int>(value) != 0);
                } else {
                    throw VMException("Type error: Expected boolean or integer for condition");
                }
                
                if (Condition) {
                    Pc_ = static_cast<size_t>(Address);
                }
                break;
            }
            
            // Memory operations
            case Opcode::LOAD: {
                size_t Address = static_cast<size_t>(PopInt());
                uint8_t value = Memory_.ReadHeap(Address);
                PushInt(value);
                break;
            }
            case Opcode::STORE: {
                size_t Address = static_cast<size_t>(PopInt());
                int value = PopInt();
                Memory_.WriteHeap(Address, static_cast<uint8_t>(value));
                break;
            }
            case Opcode::ALLOC: {
                size_t size = static_cast<size_t>(PopInt());
                size_t Address = Memory_.AllocateHeap(size);
                PushInt(static_cast<int>(Address));
                break;
            }
            
            // Miscellaneous
            case Opcode::PRINT: {
                auto value = Memory_.PopStack();
                
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
                Running_ = false;
                break;
            }
            
            default:
                throw InvalidInstructionException(static_cast<int>(Op));
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
    BytecodeAssembler& Emit(Opcode Opcode) {
        Bytecode_.push_back(static_cast<uint8_t>(Opcode));
        return *this;
    }
    
    // Add an instruction with an integer operand
    BytecodeAssembler& EmitInt(Opcode Opcode, int Operand) {
        Bytecode_.push_back(static_cast<uint8_t>(Opcode));
        
        // Little-endian format for integer operands
        Bytecode_.push_back(Operand & 0xFF);
        Bytecode_.push_back((Operand >> 8) & 0xFF);
        Bytecode_.push_back((Operand >> 16) & 0xFF);
        Bytecode_.push_back((Operand >> 24) & 0xFF);
        
        return *this;
    }
    
    // Get the current position in the bytecode (useful for jump targets)
    size_t GetCurrentPosition() const {
        return Bytecode_.size();
    }
    
    // Update a previously emitted integer operand (for forward jumps)
    void UpdateIntOperandAt(size_t Position, int NewValue) {
        if (Position + 4 >= Bytecode_.size()) {
            throw std::out_of_range("Invalid position for operand update");
        }
        
        Bytecode_[Position] = NewValue & 0xFF;
        Bytecode_[Position + 1] = (NewValue >> 8) & 0xFF;
        Bytecode_[Position + 2] = (NewValue >> 16) & 0xFF;
        Bytecode_[Position + 3] = (NewValue >> 24) & 0xFF;
    }
    
    // Get the assembled bytecode
    std::vector<uint8_t> GetBytecode() const {
        return Bytecode_;
    }
    
private:
    std::vector<uint8_t> Bytecode_;
};

/**
 * Helper functions to create test programs
 */
namespace TestPrograms {
    // Program that calculates factorial of 5
    inline std::vector<uint8_t> CreateFactorialProgram() {
        BytecodeAssembler Assembler;
        
        // Initialize variables
        Assembler.EmitInt(Opcode::PUSH, 5);     // Input number (5)
        Assembler.EmitInt(Opcode::PUSH, 1);     // Result starts at 1
        
        // Loop start position
        size_t LoopStart = Assembler.GetCurrentPosition();
        
        // Check if input <= 1
        Assembler.Emit(Opcode::DUP);           // Duplicate input for comparison
        Assembler.EmitInt(Opcode::PUSH, 1);
        Assembler.Emit(Opcode::LE);            // Check if input <= 1
        
        // Jump to end if input <= 1
        size_t JumpPos = Assembler.GetCurrentPosition();
        Assembler.EmitInt(Opcode::JZ, 0);      // Placeholder jump address, will update later
        
        // Multiply result by input
        Assembler.Emit(Opcode::DUP);           // Duplicate input
        Assembler.Emit(Opcode::SWAP);          // Swap to get result on top
        Assembler.Emit(Opcode::MUL);           // Multiply result * input
        
        // Decrement input
        Assembler.Emit(Opcode::SWAP);          // Swap to get input on top
        Assembler.EmitInt(Opcode::PUSH, 1);
        Assembler.Emit(Opcode::SUB);           // Subtract 1 from input
        
        // Jump back to loop start
        Assembler.EmitInt(Opcode::JMP, LoopStart);
        
        // Update the conditional jump address
        size_t EndPos = Assembler.GetCurrentPosition();
        Assembler.UpdateIntOperandAt(JumpPos + 1, EndPos);
        
        // Drop the input, keep only the result
        Assembler.Emit(Opcode::SWAP);
        Assembler.Emit(Opcode::POP);
        
        // Print the result
        Assembler.Emit(Opcode::PRINT);
        
        // Stop execution
        Assembler.Emit(Opcode::HALT);
        
        return Assembler.GetBytecode();
    }
    
    // Program that calculates Fibonacci sequence (first 10 numbers)
    inline std::vector<uint8_t> CreateFibonacciProgram() {
        BytecodeAssembler Assembler;
        
        // Initialize variables
        Assembler.EmitInt(Opcode::PUSH, 0);     // First number
        Assembler.Emit(Opcode::DUP);
        Assembler.Emit(Opcode::PRINT);          // Print first number (0)
        
        Assembler.EmitInt(Opcode::PUSH, 1);     // Second number
        Assembler.Emit(Opcode::DUP);
        Assembler.Emit(Opcode::PRINT);          // Print second number (1)
        
        Assembler.EmitInt(Opcode::PUSH, 8);     // Counter (8 more to print)
        
        // Loop start
        size_t LoopStart = Assembler.GetCurrentPosition();
        
        // Check if counter > 0
        Assembler.Emit(Opcode::DUP);            // Duplicate counter
        Assembler.EmitInt(Opcode::PUSH, 0);
        Assembler.Emit(Opcode::GT);             // Check if counter > 0
        
        // Jump to end if counter <= 0
        size_t JumpPos = Assembler.GetCurrentPosition();
        Assembler.EmitInt(Opcode::JZ, 0);       // Placeholder jump address
        
        // Calculate next Fibonacci number
        Assembler.Emit(Opcode::SWAP);           // Get second number
        Assembler.Emit(Opcode::DUP);            // Duplicate it
        Assembler.Emit(Opcode::SWAP);           // Swap
        Assembler.Emit(Opcode::SWAP);           // Get first number
        Assembler.Emit(Opcode::DUP);            // Duplicate it
        Assembler.Emit(Opcode::ADD);            // Add first + second
        
        // Print the next number
        Assembler.Emit(Opcode::DUP);
        Assembler.Emit(Opcode::PRINT);
        
        // Decrement counter
        Assembler.Emit(Opcode::SWAP);           // Get counter
        Assembler.EmitInt(Opcode::PUSH, 1);
        Assembler.Emit(Opcode::SUB);            // Subtract 1
        
        // Jump back to loop start
        Assembler.EmitInt(Opcode::JMP, LoopStart);
        
        // Update jump address
        size_t EndPos = Assembler.GetCurrentPosition();
        Assembler.UpdateIntOperandAt(JumpPos + 1, EndPos);
        
        // Cleanup stack and halt
        Assembler.Emit(Opcode::POP);            // Pop counter
        Assembler.Emit(Opcode::POP);            // Pop second number
        Assembler.Emit(Opcode::POP);            // Pop first number
        
        Assembler.Emit(Opcode::HALT);
        
        return Assembler.GetBytecode();
    }
    
    // Program that tests the heap memory
    inline std::vector<uint8_t> CreateHeapTestProgram() {
        BytecodeAssembler Assembler;
        
        // Allocate 10 bytes on the heap
        Assembler.EmitInt(Opcode::PUSH, 10);
        Assembler.Emit(Opcode::ALLOC);
        
        // Store the heap address for later use
        Assembler.Emit(Opcode::DUP);
        
        // Store values 1-10 in the allocated memory
        for (int i = 0; i < 10; i++) {
            // Duplicate the current address
            Assembler.Emit(Opcode::DUP);
            
            // Push the value to store (i+1)
            Assembler.EmitInt(Opcode::PUSH, i + 1);
            
            // Store value at current address
            Assembler.Emit(Opcode::STORE);
            
            // Increment address for next iteration
            if (i < 9) {  // Skip increment on last iteration
                Assembler.Emit(Opcode::DUP);
                Assembler.EmitInt(Opcode::PUSH, 1);
                Assembler.Emit(Opcode::ADD);
            }
        }
        
        // Now read back and print values
        for (int i = 0; i < 10; i++) {
            // Load value from current address
            Assembler.Emit(Opcode::LOAD);
            
            // Print the loaded value
            Assembler.Emit(Opcode::PRINT);
            
            // If not the last iteration, compute next address
            if (i < 9) {
                Assembler.EmitInt(Opcode::PUSH, i + 1);
                Assembler.Emit(Opcode::ADD);
            }
        }
        
        Assembler.Emit(Opcode::HALT);
        
        return Assembler.GetBytecode();
    }
}

/**
 * Main test function to demonstrate VM execution
 */
