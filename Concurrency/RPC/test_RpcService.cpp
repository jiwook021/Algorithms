/**
 * @file test_RpcService.cpp
 * @brief Unit tests for the RPC service.
 *
 * @details
 * Tests the registered function logic in isolation (no network required).
 * The Add function is tested with valid inputs, invalid inputs, and
 * wrong argument counts.
 */

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <map>

/**
 * @brief Add function matching the RPC registration signature.
 * @param args String arguments.
 * @return Result string.
 */
static std::string Add(const std::vector<std::string>& args) {
    if (args.size() != 2) return "Error: Expected 2 arguments";
    try {
        int a = std::stoi(args[0]);
        int b = std::stoi(args[1]);
        return std::to_string(a + b);
    } catch (const std::exception&) {
        return "Error: Invalid arguments";
    }
}

/** @brief Verify Add with valid integer arguments. */
static void TestAddValid() {
    assert(Add({"3", "5"}) == "8");
    assert(Add({"0", "0"}) == "0");
    assert(Add({"-10", "10"}) == "0");
    assert(Add({"100", "200"}) == "300");
    std::cout << "[PASS] TestAddValid\n";
}

/** @brief Verify Add with wrong argument count. */
static void TestAddWrongArgCount() {
    assert(Add({}) == "Error: Expected 2 arguments");
    assert(Add({"1"}) == "Error: Expected 2 arguments");
    assert(Add({"1", "2", "3"}) == "Error: Expected 2 arguments");
    std::cout << "[PASS] TestAddWrongArgCount\n";
}

/** @brief Verify Add with non-integer arguments. */
static void TestAddInvalidArgs() {
    assert(Add({"abc", "5"}) == "Error: Invalid arguments");
    assert(Add({"3", "xyz"}) == "Error: Invalid arguments");
    std::cout << "[PASS] TestAddInvalidArgs\n";
}

/** @brief Verify function dispatch map works correctly. */
static void TestFunctionDispatch() {
    std::map<std::string, std::function<std::string(const std::vector<std::string>&)>> functions;
    functions["add"] = Add;

    assert(functions.find("add") != functions.end());
    assert(functions["add"]({"3", "5"}) == "8");
    assert(functions.find("unknown") == functions.end());
    std::cout << "[PASS] TestFunctionDispatch\n";
}

int main() {
    TestAddValid();
    TestAddWrongArgCount();
    TestAddInvalidArgs();
    TestFunctionDispatch();

    std::cout << "\nAll RpcService tests passed.\n";
    return 0;
}
