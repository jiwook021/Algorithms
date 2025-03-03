/**
 * @file test_MqttCommon.cpp
 * @brief Unit tests for MQTT encoding/decoding utilities.
 *
 * Tests the pure functions (no network required).
 */

#include "MqttCommon.hpp"
#include <cassert>
#include <iostream>

/// EncodeRemainingLength for small values (single byte).
void TestEncodeSmallLength() {
    auto encoded = Mqtt::EncodeRemainingLength(0);
    assert(encoded.size() == 1);
    assert(encoded[0] == 0x00);

    encoded = Mqtt::EncodeRemainingLength(127);
    assert(encoded.size() == 1);
    assert(encoded[0] == 127);

    std::cout << "[PASS] TestEncodeSmallLength\n";
}

/// EncodeRemainingLength for values requiring 2 bytes.
void TestEncodeTwoByteLength() {
    auto encoded = Mqtt::EncodeRemainingLength(128);
    assert(encoded.size() == 2);
    assert(encoded[0] == 0x80);
    assert(encoded[1] == 0x01);

    encoded = Mqtt::EncodeRemainingLength(16383);
    assert(encoded.size() == 2);
    assert(encoded[0] == 0xFF);
    assert(encoded[1] == 0x7F);

    std::cout << "[PASS] TestEncodeTwoByteLength\n";
}

/// Round-trip: encode then decode from buffer.
void TestEncodeDecodeRoundTrip() {
    for (size_t val : {0, 1, 127, 128, 255, 16383, 16384, 2097151, 268435455}) {
        auto encoded = Mqtt::EncodeRemainingLength(val);
        size_t consumed = 0;
        size_t decoded = Mqtt::DecodeRemainingLengthFromBuffer(encoded, 0, consumed);
        assert(decoded == val);
        assert(consumed == encoded.size());
    }

    std::cout << "[PASS] TestEncodeDecodeRoundTrip\n";
}

/// EncodeString produces 2-byte length prefix + payload.
void TestEncodeString() {
    auto encoded = Mqtt::EncodeString("MQTT");
    assert(encoded.size() == 6); // 2 byte len + 4 chars
    assert(encoded[0] == 0x00);
    assert(encoded[1] == 0x04);
    assert(encoded[2] == 'M');
    assert(encoded[3] == 'Q');
    assert(encoded[4] == 'T');
    assert(encoded[5] == 'T');

    std::cout << "[PASS] TestEncodeString\n";
}

/// EncodeString for empty string.
void TestEncodeEmptyString() {
    auto encoded = Mqtt::EncodeString("");
    assert(encoded.size() == 2);
    assert(encoded[0] == 0x00);
    assert(encoded[1] == 0x00);

    std::cout << "[PASS] TestEncodeEmptyString\n";
}

/// EncodeString for longer string.
void TestEncodeLongerString() {
    std::string s(300, 'x');
    auto encoded = Mqtt::EncodeString(s);
    assert(encoded.size() == 302);
    // Length = 300 = 0x012C
    assert(encoded[0] == 0x01);
    assert(encoded[1] == 0x2C);

    std::cout << "[PASS] TestEncodeLongerString\n";
}

int main() {
    TestEncodeSmallLength();
    TestEncodeTwoByteLength();
    TestEncodeDecodeRoundTrip();
    TestEncodeString();
    TestEncodeEmptyString();
    TestEncodeLongerString();
    std::cout << "\nAll MqttCommon tests passed.\n";
    return 0;
}
