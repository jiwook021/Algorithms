/**
 * @file MqttCommon.hpp
 * @brief MQTT v3.1.1 protocol helpers and base class
 * @details Provides packet encoding/decoding utilities and a MqttBase class
 *          for connecting to an MQTT broker, used by publisher and subscriber.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

namespace Mqtt {

/// Send all bytes in buf or throw.
void SendAll(int sockFd, const std::vector<uint8_t>& buf);

/// Receive exactly len bytes or throw.
std::vector<uint8_t> RecvAll(int sockFd, size_t len);

/// Encode an MQTT remaining-length field (variable-length encoding).
std::vector<uint8_t> EncodeRemainingLength(size_t length);

/// Decode an MQTT remaining-length field from a byte buffer.
/// Returns the decoded value and sets bytesConsumed.
size_t DecodeRemainingLengthFromBuffer(const std::vector<uint8_t>& data,
                                       size_t offset,
                                       size_t& bytesConsumed);

/// Decode remaining length by reading from a socket.
size_t DecodeRemainingLength(int sockFd);

/// Encode a UTF-8 string with 2-byte length prefix (MQTT format).
std::vector<uint8_t> EncodeString(const std::string& s);

/// Base class for common MQTT operations (connect / disconnect).
class MqttBase {
public:
    MqttBase(const std::string& host, uint16_t port, const std::string& clientId);
    virtual ~MqttBase();

    /// Perform MQTT CONNECT handshake.
    void ConnectToBroker(int keepAliveSec = 60);

    /// Send DISCONNECT and close socket.
    void Disconnect();

protected:
    std::string host_;
    uint16_t port_;
    std::string clientId_;
    int sockFd_;
};

} // namespace Mqtt
