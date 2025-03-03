/**
 * @file test_VirtualMachine.cpp
 * @brief Unit tests for the VirtualMachine module.
 */

#include "VirtualMachine.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

/// Test basic PUSH / POP / stack operations.
void TestPushPop() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.EmitInt(Opcode::PUSH, 42);
    assembler.EmitInt(Opcode::PUSH, 99);
    assembler.Emit(Opcode::POP);       // remove 99
    assembler.Emit(Opcode::PRINT);     // print 42
    assembler.Emit(Opcode::HALT);

    // Redirect cout to verify output
    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());

    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();

    std::cout.rdbuf(oldBuf);
    assert(oss.str() == "42\n");
    std::cout << "[PASS] TestPushPop\n";
}

/// Test ADD instruction.
void TestAdd() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.EmitInt(Opcode::PUSH, 10);
    assembler.EmitInt(Opcode::PUSH, 20);
    assembler.Emit(Opcode::ADD);
    assembler.Emit(Opcode::PRINT);
    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "30\n");
    std::cout << "[PASS] TestAdd\n";
}

/// Test SUB instruction.
void TestSub() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.EmitInt(Opcode::PUSH, 50);
    assembler.EmitInt(Opcode::PUSH, 8);
    assembler.Emit(Opcode::SUB);
    assembler.Emit(Opcode::PRINT);
    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "42\n");
    std::cout << "[PASS] TestSub\n";
}

/// Test MUL instruction.
void TestMul() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.EmitInt(Opcode::PUSH, 6);
    assembler.EmitInt(Opcode::PUSH, 7);
    assembler.Emit(Opcode::MUL);
    assembler.Emit(Opcode::PRINT);
    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "42\n");
    std::cout << "[PASS] TestMul\n";
}

/// Test DIV instruction and division-by-zero exception.
void TestDiv() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.EmitInt(Opcode::PUSH, 100);
    assembler.EmitInt(Opcode::PUSH, 4);
    assembler.Emit(Opcode::DIV);
    assembler.Emit(Opcode::PRINT);
    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "25\n");

    // Test division by zero
    BytecodeAssembler assembler2;
    assembler2.EmitInt(Opcode::PUSH, 10);
    assembler2.EmitInt(Opcode::PUSH, 0);
    assembler2.Emit(Opcode::DIV);
    assembler2.Emit(Opcode::HALT);

    bool caught = false;
    try {
        vm.LoadProgram(assembler2.GetBytecode());
        vm.Run();
    } catch (const DivisionByZeroException&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestDiv\n";
}

/// Test DUP and SWAP instructions.
void TestDupSwap() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    // Push 5, DUP -> stack: [5, 5], ADD -> 10
    assembler.EmitInt(Opcode::PUSH, 5);
    assembler.Emit(Opcode::DUP);
    assembler.Emit(Opcode::ADD);
    assembler.Emit(Opcode::PRINT);

    // Push 1, Push 2, SWAP -> [2,1], SUB -> 2-1=1
    assembler.EmitInt(Opcode::PUSH, 1);
    assembler.EmitInt(Opcode::PUSH, 2);
    assembler.Emit(Opcode::SWAP);
    assembler.Emit(Opcode::SUB);
    assembler.Emit(Opcode::PRINT);

    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "10\n1\n");
    std::cout << "[PASS] TestDupSwap\n";
}

/// Test comparison operations.
void TestComparisons() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    // 5 == 5 -> true
    assembler.EmitInt(Opcode::PUSH, 5);
    assembler.EmitInt(Opcode::PUSH, 5);
    assembler.Emit(Opcode::EQ);
    assembler.Emit(Opcode::PRINT);

    // 3 < 5 -> true
    assembler.EmitInt(Opcode::PUSH, 3);
    assembler.EmitInt(Opcode::PUSH, 5);
    assembler.Emit(Opcode::LT);
    assembler.Emit(Opcode::PRINT);

    assembler.Emit(Opcode::HALT);

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(assembler.GetBytecode());
    vm.Run();
    std::cout.rdbuf(oldBuf);

    assert(oss.str() == "true\ntrue\n");
    std::cout << "[PASS] TestComparisons\n";
}

/// Test stack underflow exception.
void TestStackUnderflow() {
    VirtualMachine vm;
    BytecodeAssembler assembler;

    assembler.Emit(Opcode::POP);
    assembler.Emit(Opcode::HALT);

    bool caught = false;
    try {
        vm.LoadProgram(assembler.GetBytecode());
        vm.Run();
    } catch (const StackUnderflowException&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestStackUnderflow\n";
}

/// Test that the factorial program runs without crashing and produces output.
void TestFactorialProgram() {
    VirtualMachine vm;
    auto program = TestPrograms::CreateFactorialProgram();

    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf(oss.rdbuf());
    vm.LoadProgram(program);
    vm.Run();
    std::cout.rdbuf(oldBuf);

    // The factorial bytecode runs and produces integer output.
    assert(!oss.str().empty());
    std::cout << "[PASS] TestFactorialProgram\n";
}

int main() {
    TestPushPop();
    TestAdd();
    TestSub();
    TestMul();
    TestDiv();
    TestDupSwap();
    TestComparisons();
    TestStackUnderflow();
    TestFactorialProgram();
    std::cout << "\nAll VirtualMachine tests passed.\n";
    return 0;
}
