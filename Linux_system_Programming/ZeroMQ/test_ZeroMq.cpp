#include "ZeroMq.hpp"
#include <cassert>
#include <iostream>

using namespace ZeroMq;

// ============================================================================
// Endpoint Validation Tests
// ============================================================================

void TestValidateEndpointTcp() {
    assert(ValidateEndpoint("tcp://*:5555") == true);
    assert(ValidateEndpoint("tcp://localhost:5555") == true);
    assert(ValidateEndpoint("tcp://192.168.1.1:9000") == true);
    std::cout << "[PASS] TestValidateEndpointTcp" << std::endl;
}

void TestValidateEndpointIpc() {
    assert(ValidateEndpoint("ipc:///tmp/zmq.sock") == true);
    std::cout << "[PASS] TestValidateEndpointIpc" << std::endl;
}

void TestValidateEndpointInproc() {
    assert(ValidateEndpoint("inproc://my-channel") == true);
    std::cout << "[PASS] TestValidateEndpointInproc" << std::endl;
}

void TestValidateEndpointInvalid() {
    assert(ValidateEndpoint("") == false);
    assert(ValidateEndpoint("http://localhost") == false);
    assert(ValidateEndpoint("tcp://") == false);  // nothing after prefix
    assert(ValidateEndpoint("foobar") == false);
    std::cout << "[PASS] TestValidateEndpointInvalid" << std::endl;
}

void TestExtractPortValid() {
    auto port = ExtractPort("tcp://localhost:5555");
    assert(port.has_value());
    assert(*port == 5555);
    std::cout << "[PASS] TestExtractPortValid" << std::endl;
}

void TestExtractPortWildcard() {
    auto port = ExtractPort("tcp://*:9999");
    assert(port.has_value());
    assert(*port == 9999);
    std::cout << "[PASS] TestExtractPortWildcard" << std::endl;
}

void TestExtractPortNonTcp() {
    auto port = ExtractPort("ipc:///tmp/zmq.sock");
    assert(!port.has_value());
    std::cout << "[PASS] TestExtractPortNonTcp" << std::endl;
}

void TestExtractPortInvalid() {
    auto port = ExtractPort("tcp://localhost:abc");
    assert(!port.has_value());
    std::cout << "[PASS] TestExtractPortInvalid" << std::endl;
}

void TestExtractPortOutOfRange() {
    auto port = ExtractPort("tcp://localhost:99999");
    assert(!port.has_value());
    std::cout << "[PASS] TestExtractPortOutOfRange" << std::endl;
}

void TestExtractTransport() {
    assert(ExtractTransport("tcp://localhost:5555") == "tcp");
    assert(ExtractTransport("ipc:///tmp/sock") == "ipc");
    assert(ExtractTransport("inproc://channel") == "inproc");
    assert(ExtractTransport("no-transport") == "");
    std::cout << "[PASS] TestExtractTransport" << std::endl;
}

// ============================================================================
// Message Formatting Tests
// ============================================================================

void TestFormatMessage() {
    assert(FormatMessage(0) == "Hello ZeroMQ 0");
    assert(FormatMessage(42) == "Hello ZeroMQ 42");
    assert(FormatMessage(100) == "Hello ZeroMQ 100");
    std::cout << "[PASS] TestFormatMessage" << std::endl;
}

void TestParseMessageCountValid() {
    auto count = ParseMessageCount("Hello ZeroMQ 7");
    assert(count.has_value());
    assert(*count == 7);
    std::cout << "[PASS] TestParseMessageCountValid" << std::endl;
}

void TestParseMessageCountInvalid() {
    assert(!ParseMessageCount("Not a zmq message").has_value());
    assert(!ParseMessageCount("Hello ZeroMQ ").has_value());
    assert(!ParseMessageCount("").has_value());
    std::cout << "[PASS] TestParseMessageCountInvalid" << std::endl;
}

void TestFormatParseRoundTrip() {
    for (int i = 0; i < 10; ++i) {
        std::string msg = FormatMessage(i);
        auto parsed = ParseMessageCount(msg);
        assert(parsed.has_value());
        assert(*parsed == i);
    }
    std::cout << "[PASS] TestFormatParseRoundTrip" << std::endl;
}

// ============================================================================
// Topic Message Tests
// ============================================================================

void TestBuildTopicMessage() {
    std::string msg = BuildTopicMessage("weather", "temp=22");
    assert(msg == "weather temp=22");
    std::cout << "[PASS] TestBuildTopicMessage" << std::endl;
}

void TestSplitTopicMessageValid() {
    auto result = SplitTopicMessage("stocks AAPL=150.5");
    assert(result.has_value());
    assert(result->topic == "stocks");
    assert(result->payload == "AAPL=150.5");
    std::cout << "[PASS] TestSplitTopicMessageValid" << std::endl;
}

void TestSplitTopicMessageNoSpace() {
    auto result = SplitTopicMessage("nospacemessage");
    assert(!result.has_value());
    std::cout << "[PASS] TestSplitTopicMessageNoSpace" << std::endl;
}

void TestSplitTopicMessageLeadingSpace() {
    auto result = SplitTopicMessage(" leadingspace");
    assert(!result.has_value());
    std::cout << "[PASS] TestSplitTopicMessageLeadingSpace" << std::endl;
}

void TestBuildSplitRoundTrip() {
    std::string msg = BuildTopicMessage("alerts", "fire detected");
    auto split = SplitTopicMessage(msg);
    assert(split.has_value());
    assert(split->topic == "alerts");
    assert(split->payload == "fire detected");
    std::cout << "[PASS] TestBuildSplitRoundTrip" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== ZeroMq Unit Tests ===" << std::endl;

    TestValidateEndpointTcp();
    TestValidateEndpointIpc();
    TestValidateEndpointInproc();
    TestValidateEndpointInvalid();
    TestExtractPortValid();
    TestExtractPortWildcard();
    TestExtractPortNonTcp();
    TestExtractPortInvalid();
    TestExtractPortOutOfRange();
    TestExtractTransport();

    TestFormatMessage();
    TestParseMessageCountValid();
    TestParseMessageCountInvalid();
    TestFormatParseRoundTrip();

    TestBuildTopicMessage();
    TestSplitTopicMessageValid();
    TestSplitTopicMessageNoSpace();
    TestSplitTopicMessageLeadingSpace();
    TestBuildSplitRoundTrip();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
