#include "ZeroMq.hpp"
#include <sstream>
#include <algorithm>

namespace ZeroMq {

// ============================================================================
// Endpoint validation
// ============================================================================

static const std::vector<std::string> VALID_TRANSPORTS = {
    "tcp://", "ipc://", "inproc://", "pgm://", "epgm://"
};

bool ValidateEndpoint(const std::string& endpoint) {
    if (endpoint.empty()) return false;
    for (const auto& transport : VALID_TRANSPORTS) {
        if (endpoint.compare(0, transport.size(), transport) == 0) {
            // Must have something after the transport prefix.
            return endpoint.size() > transport.size();
        }
    }
    return false;
}

std::optional<int> ExtractPort(const std::string& endpoint) {
    if (endpoint.compare(0, 6, "tcp://") != 0) return std::nullopt;

    size_t colonPos = endpoint.rfind(':');
    // The last colon must be after "tcp://" (position >= 6).
    if (colonPos == std::string::npos || colonPos < 6) return std::nullopt;

    std::string portStr = endpoint.substr(colonPos + 1);
    if (portStr.empty()) return std::nullopt;

    try {
        int port = std::stoi(portStr);
        if (port < 0 || port > 65535) return std::nullopt;
        return port;
    } catch (...) {
        return std::nullopt;
    }
}

std::string ExtractTransport(const std::string& endpoint) {
    size_t pos = endpoint.find("://");
    if (pos == std::string::npos) return "";
    return endpoint.substr(0, pos);
}

// ============================================================================
// Message formatting
// ============================================================================

std::string FormatMessage(int count) {
    return "Hello ZeroMQ " + std::to_string(count);
}

std::optional<int> ParseMessageCount(const std::string& message) {
    const std::string PREFIX = "Hello ZeroMQ ";
    if (message.compare(0, PREFIX.size(), PREFIX) != 0) return std::nullopt;

    std::string countStr = message.substr(PREFIX.size());
    if (countStr.empty()) return std::nullopt;

    try {
        return std::stoi(countStr);
    } catch (...) {
        return std::nullopt;
    }
}

std::string BuildTopicMessage(const std::string& topic, const std::string& payload) {
    return topic + " " + payload;
}

std::optional<TopicMessage> SplitTopicMessage(const std::string& message) {
    size_t spacePos = message.find(' ');
    if (spacePos == std::string::npos || spacePos == 0) return std::nullopt;

    TopicMessage result;
    result.topic = message.substr(0, spacePos);
    result.payload = message.substr(spacePos + 1);
    return result;
}

} // namespace ZeroMq
