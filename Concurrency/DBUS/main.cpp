/**
 * @file main.cpp
 * @brief Driver program for the D-Bus Calculator service.
 *
 * Registers the Calculator service on the session bus and enters the
 * event loop to handle incoming method calls.
 *
 * Build:  make && ./main
 */

#include "DbusCalculator.hpp"

int main() {
    DBus::BusDispatcher dispatcher;
    DBus::default_dispatcher = &dispatcher;

    DBus::Connection conn = DBus::Connection::SessionBus();

    try {
        conn.RequestName("com.example.Calculator");
    } catch (DBus::Error& e) {
        std::cerr << "Failed to request bus name: " << e.what() << std::endl;
        return 1;
    }

    DbusCalculator calc(conn);

    std::cout << "Calculator DBus service is running. "
              << "Use a DBus client to call 'Add' or 'Subtract' methods."
              << std::endl;

    dispatcher.Enter();
    return 0;
}
