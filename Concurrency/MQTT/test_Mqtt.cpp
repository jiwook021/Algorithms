/**
 * @file test_Mqtt.cpp
 * @brief Unit tests for the Mqtt module.
 *
 * @details
 * Tests cover Message construction, ConnectOptions defaults, broker address
 * parsing, and QoS validation. Network-dependent tests are not included
 * since they require a running broker.
 */

#include "Mqtt.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

/** @brief Verify Message construction with valid QoS levels. */
static void TestMessageConstruction() {
    Mqtt::Message msg("sensor/temp", "23.5", 0, false);
    assert(msg.Topic == "sensor/temp");
    assert(msg.Payload == "23.5");
    assert(msg.Qos == 0);
    assert(msg.Retained == false);

    Mqtt::Message msg1("topic", "data", 1, true);
    assert(msg1.Qos == 1);
    assert(msg1.Retained == true);

    Mqtt::Message msg2("topic", "data", 2, false);
    assert(msg2.Qos == 2);

    std::cout << "[PASS] TestMessageConstruction\n";
}

/** @brief Verify Message throws on invalid QoS. */
static void TestMessageInvalidQos() {
    bool caught = false;
    try {
        Mqtt::Message msg("topic", "data", 3, false);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        Mqtt::Message msg("topic", "data", -1, false);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "[PASS] TestMessageInvalidQos\n";
}

/** @brief Verify ConnectOptions default values. */
static void TestConnectOptionsDefaults() {
    Mqtt::ConnectOptions opts;
    assert(opts.BrokerAddress == "tcp://localhost:1883");
    assert(opts.ClientId == "default_client");
    assert(opts.KeepAliveInterval == 20);
    std::cout << "[PASS] TestConnectOptionsDefaults\n";
}

/** @brief Verify AsyncClient parses valid broker URIs. */
static void TestAsyncClientValidUri() {
    // Should not throw for valid TCP URIs
    Mqtt::AsyncClient client1("tcp://localhost:1883", "c1");
    Mqtt::AsyncClient client2("tcp://192.168.1.1:9999", "c2");
    Mqtt::AsyncClient client3("tcp://broker.example.com:1883", "c3");
    std::cout << "[PASS] TestAsyncClientValidUri\n";
}

/** @brief Verify AsyncClient throws on invalid protocol. */
static void TestAsyncClientInvalidProtocol() {
    bool caught = false;
    try {
        Mqtt::AsyncClient client("http://localhost:1883", "c1");
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestAsyncClientInvalidProtocol\n";
}

/** @brief Verify AsyncClient throws on malformed URI. */
static void TestAsyncClientMalformedUri() {
    bool caught = false;
    try {
        Mqtt::AsyncClient client("not-a-uri", "c1");
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] TestAsyncClientMalformedUri\n";
}

/** @brief Verify EncodeRemainingLength for various values. */
static void TestEncodeRemainingLength() {
    auto r1 = EncodeRemainingLength(0);
    assert(r1.size() == 1 && r1[0] == 0);

    auto r2 = EncodeRemainingLength(127);
    assert(r2.size() == 1 && r2[0] == 127);

    auto r3 = EncodeRemainingLength(128);
    assert(r3.size() == 2 && r3[0] == 0x80 && r3[1] == 0x01);

    auto r4 = EncodeRemainingLength(16383);
    assert(r4.size() == 2 && r4[0] == 0xFF && r4[1] == 0x7F);

    std::cout << "[PASS] TestEncodeRemainingLength\n";
}

int main() {
    TestMessageConstruction();
    TestMessageInvalidQos();
    TestConnectOptionsDefaults();
    TestAsyncClientValidUri();
    TestAsyncClientInvalidProtocol();
    TestAsyncClientMalformedUri();
    TestEncodeRemainingLength();

    std::cout << "\nAll Mqtt tests passed.\n";
    return 0;
}
