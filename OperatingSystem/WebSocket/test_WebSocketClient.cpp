/**
 * @file test_WebSocketClient.cpp
 * @brief Unit tests for WebSocketClient.
 *
 * Since WebSocket requires a live server, these tests verify that the
 * Client class can be constructed and that error handling works correctly
 * when no server is available.
 */

#include "WebSocketClient.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

/// Client can be default-constructed without crashing.
void TestClientConstruction() {
    WebSocket::Client client;
    // Just verify it constructs without error.
    std::cout << "[PASS] TestClientConstruction\n";
}

/// Connecting to an invalid host throws.
void TestConnectToInvalidHost() {
    WebSocket::Client client;
    bool caught = false;
    try {
        client.Connect("this.host.does.not.exist.invalid", "12345");
    } catch (const std::exception&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestConnectToInvalidHost\n";
}

/// Connecting to a port with no server throws.
void TestConnectRefused() {
    WebSocket::Client client;
    bool caught = false;
    try {
        // Port 1 is unlikely to have a WebSocket server.
        client.Connect("127.0.0.1", "1");
    } catch (const std::exception&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestConnectRefused\n";
}

int main() {
    TestClientConstruction();
    TestConnectToInvalidHost();
    TestConnectRefused();
    std::cout << "\nAll WebSocketClient tests passed.\n";
    return 0;
}
