/**
 * @file DbusCalculator.hpp
 * @brief DBus Calculator service -- exposes Add/Subtract over the session bus.
 *
 * @details
 * Uses dbus-c++ to register a Calculator service at /com/example/Calculator
 * on the D-Bus session bus. Clients can call Add() and Subtract() remotely.
 *
 * @note Requires libdbus-c++-1 to compile.
 */

#ifndef DBUS_CALCULATOR_HPP
#define DBUS_CALCULATOR_HPP

#include <dbus-c++-1/dbus-c++/dbus.h>
#include <string>
#include <iostream>
#include <cstdint>

/**
 * @class DbusCalculator
 * @brief D-Bus service exposing arithmetic operations.
 */
class DbusCalculator : public DBus::ObjectAdaptor {
public:
    /**
     * @brief Register the calculator object on the given connection.
     * @param connection Active D-Bus connection.
     */
    DbusCalculator(DBus::Connection& connection)
        : DBus::ObjectAdaptor(connection, "/com/example/Calculator") {}

    /**
     * @brief Add two integers.
     * @param a First operand.
     * @param b Second operand.
     * @return Sum of a and b.
     */
    int32_t Add(const int32_t& a, const int32_t& b) {
        return a + b;
    }

    /**
     * @brief Subtract two integers.
     * @param a First operand.
     * @param b Second operand.
     * @return Difference (a - b).
     */
    int32_t Subtract(const int32_t& a, const int32_t& b) {
        return a - b;
    }
};

#endif // DBUS_CALCULATOR_HPP
