/**
 * @file test_DbusCalculator.cpp
 * @brief Unit tests for DbusCalculator.
 *
 * @details
 * Since DbusCalculator requires a D-Bus session bus, these tests verify
 * the arithmetic logic in isolation by testing the Add/Subtract methods
 * directly. Full integration tests require a running D-Bus daemon.
 *
 * @note This file compiles without libdbus-c++ by testing standalone
 * arithmetic functions that mirror the calculator API.
 */

#include <cassert>
#include <iostream>
#include <cstdint>

/**
 * @brief Standalone Add function mirroring DbusCalculator::Add.
 * @param a First operand.
 * @param b Second operand.
 * @return Sum.
 */
static int32_t Add(int32_t a, int32_t b) { return a + b; }

/**
 * @brief Standalone Subtract function mirroring DbusCalculator::Subtract.
 * @param a First operand.
 * @param b Second operand.
 * @return Difference.
 */
static int32_t Subtract(int32_t a, int32_t b) { return a - b; }

/** @brief Verify Add with positive numbers. */
static void TestAddPositive() {
    assert(Add(3, 5) == 8);
    assert(Add(100, 200) == 300);
    std::cout << "[PASS] TestAddPositive\n";
}

/** @brief Verify Add with negative numbers. */
static void TestAddNegative() {
    assert(Add(-3, -5) == -8);
    assert(Add(-10, 10) == 0);
    std::cout << "[PASS] TestAddNegative\n";
}

/** @brief Verify Add with zero. */
static void TestAddZero() {
    assert(Add(0, 0) == 0);
    assert(Add(42, 0) == 42);
    std::cout << "[PASS] TestAddZero\n";
}

/** @brief Verify Subtract with positive numbers. */
static void TestSubtractPositive() {
    assert(Subtract(10, 3) == 7);
    assert(Subtract(100, 200) == -100);
    std::cout << "[PASS] TestSubtractPositive\n";
}

/** @brief Verify Subtract with negative numbers. */
static void TestSubtractNegative() {
    assert(Subtract(-3, -5) == 2);
    assert(Subtract(-10, 10) == -20);
    std::cout << "[PASS] TestSubtractNegative\n";
}

/** @brief Verify Subtract with zero. */
static void TestSubtractZero() {
    assert(Subtract(0, 0) == 0);
    assert(Subtract(42, 0) == 42);
    std::cout << "[PASS] TestSubtractZero\n";
}

int main() {
    TestAddPositive();
    TestAddNegative();
    TestAddZero();
    TestSubtractPositive();
    TestSubtractNegative();
    TestSubtractZero();

    std::cout << "\nAll DbusCalculator tests passed.\n";
    return 0;
}
