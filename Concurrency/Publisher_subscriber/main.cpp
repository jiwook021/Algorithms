/**
 * @file main.cpp
 * @brief Driver program for the Publisher-Subscriber pattern demo.
 *
 * Demonstrates callback-based subscribing, notifying, and unsubscribing
 * with the templatized Publisher.
 *
 * Build:  make && ./main
 */

#include "PublisherSubscriber.hpp"
#include <iostream>
#include <string>

int main() {
    Publisher<std::string> publisher;

    // Subscribe two callbacks simulating console and file logging.
    auto console_id = publisher.Subscribe([](const std::string& msg) {
        std::cout << "Console: " << msg << std::endl;
    });

    auto file_id = publisher.Subscribe([](const std::string& msg) {
        std::cout << "File: " << msg << std::endl;
    });

    std::cout << "First notification:\n";
    publisher.Notify("Event occurred!");

    publisher.Unsubscribe(file_id);

    std::cout << "\nSecond notification (after unsubscribing file logger):\n";
    publisher.Notify("Another event occurred!");

    // Demonstrate with a non-string message type.
    Publisher<int> int_publisher;
    auto int_id = int_publisher.Subscribe([](const int& value) {
        std::cout << "Received int: " << value << std::endl;
    });

    std::cout << "\nInteger notification:\n";
    int_publisher.Notify(42);

    // Suppress unused-variable warning.
    (void)console_id;
    (void)int_id;

    return 0;
}
