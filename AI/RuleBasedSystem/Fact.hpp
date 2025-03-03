/**
 * @file Fact.hpp
 * @brief Represents a single fact in the knowledge base
 * @details A fact is a named value that can hold bool, int, double, or string.
 *          Uses the C++20 Overloaded lambda pattern for std::visit dispatch.
 */

#pragma once

#include <string>
#include <variant>

/**
 * @brief C++20 overloaded-lambda helper for std::visit
 *
 * Allows composing multiple lambdas into one visitor:
 *   std::visit(Overloaded{
 *       [](int v)  { ... },
 *       [](auto v) { ... }
 *   }, variant);
 */
template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

/**
 * @brief Represents a fact in the knowledge base
 *
 * Facts can be of different types, using std::variant to support multiple types.
 */
class Fact {
public:
    /// Supported fact value types: boolean, integer, double, and string
    using FactValue = std::variant<bool, int, double, std::string>;

    /**
     * @brief Construct a fact with a name and value
     * @param name The name of the fact
     * @param value The value of the fact
     */
    Fact(std::string name, FactValue value)
        : name_(std::move(name)), value_(std::move(value)) {}

    /**
     * @brief Get the name of the fact
     * @return The fact name
     */
    [[nodiscard]] const std::string& GetName() const {
        return name_;
    }

    /**
     * @brief Get the value of the fact
     * @return The fact value
     */
    [[nodiscard]] const FactValue& GetValue() const {
        return value_;
    }

    /**
     * @brief Update the fact value
     * @param value The new value
     */
    void SetValue(FactValue value) {
        value_ = std::move(value);
    }

    /**
     * @brief Convert fact to string representation
     * @return String representation of the fact
     */
    [[nodiscard]] std::string ToString() const {
        std::string result = name_ + " = ";

        std::visit(Overloaded{
            [&result](bool val) {
                result += val ? "true" : "false";
            },
            [&result](int val) {
                result += std::to_string(val);
            },
            [&result](double val) {
                result += std::to_string(val);
            },
            [&result](const std::string& val) {
                result += "\"" + val + "\"";
            }
        }, value_);

        return result;
    }

private:
    std::string name_;
    FactValue value_;
};
