/**
 * @file main.cpp
 * @brief Entry point for ZeroMQ PUB-SUB demo.
 *
 * Demonstrates the testable helpers from ZeroMq.hpp.
 * The actual publisher/subscriber executables using libzmq are in
 * publisher.cpp and subscriber.cpp (retained for reference).
 */

#include "ZeroMq.hpp"
#include <iostream>

int main() {
    std::cout << "ZeroMQ Helper Demo" << std::endl;
    std::cout << "==================" << std::endl;

    // Endpoint validation
    std::string endpoint = "tcp://*:5555";
    std::cout << "Endpoint '" << endpoint << "' valid: "
              << (ZeroMq::ValidateEndpoint(endpoint) ? "yes" : "no") << std::endl;

    auto port = ZeroMq::ExtractPort("tcp://localhost:5555");
    if (port) std::cout << "Extracted port: " << *port << std::endl;

    std::cout << "Transport: " << ZeroMq::ExtractTransport(endpoint) << std::endl;

    // Message formatting
    for (int i = 0; i < 5; ++i) {
        std::string msg = ZeroMq::FormatMessage(i);
        std::cout << "Formatted: " << msg << std::endl;

        auto parsed = ZeroMq::ParseMessageCount(msg);
        if (parsed) std::cout << "  Parsed count: " << *parsed << std::endl;
    }

    // Topic messages
    std::string topicMsg = ZeroMq::BuildTopicMessage("weather", "temperature=22");
    std::cout << "Topic message: " << topicMsg << std::endl;

    auto split = ZeroMq::SplitTopicMessage(topicMsg);
    if (split) {
        std::cout << "  Topic: " << split->topic << std::endl;
        std::cout << "  Payload: " << split->payload << std::endl;
    }

    return 0;
}
