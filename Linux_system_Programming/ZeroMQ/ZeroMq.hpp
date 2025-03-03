#pragma once

/**
 * @file ZeroMq.hpp
 * @brief ZeroMQ PUB-SUB messaging pattern wrapper.
 *
 * Provides a Publisher and Subscriber abstraction over the zmq.hpp
 * (cppzmq) C++ bindings. Also exposes testable pure-logic helpers
 * for endpoint validation and message formatting.
 *
 * Requires: libzmq3-dev and cppzmq headers (sudo apt install libzmq3-dev).
 */

#include <string>
#include <optional>
#include <vector>

namespace ZeroMq {

// ============================================================================
// Endpoint validation (testable without libzmq)
// ============================================================================

/// Validate a ZeroMQ endpoint string (e.g. "tcp://*:5555", "ipc:///tmp/feed").
/// Must start with a known transport ("tcp://", "ipc://", "inproc://", "pgm://", "epgm://").
bool ValidateEndpoint(const std::string& endpoint);

/// Extract the port number from a tcp:// endpoint. Returns nullopt if not tcp or invalid.
std::optional<int> ExtractPort(const std::string& endpoint);

/// Extract the transport protocol from an endpoint (e.g. "tcp", "ipc").
std::string ExtractTransport(const std::string& endpoint);

// ============================================================================
// Message formatting (testable without libzmq)
// ============================================================================

/// Format a numbered message: "Hello ZeroMQ <count>".
std::string FormatMessage(int count);

/// Parse a numbered message and extract the count. Returns nullopt on failure.
std::optional<int> ParseMessageCount(const std::string& message);

/// Build a multi-part topic message: "<topic> <payload>".
std::string BuildTopicMessage(const std::string& topic, const std::string& payload);

/// Split a topic message into (topic, payload). Returns nullopt if no space found.
struct TopicMessage {
    std::string topic;
    std::string payload;
};
std::optional<TopicMessage> SplitTopicMessage(const std::string& message);

} // namespace ZeroMq
