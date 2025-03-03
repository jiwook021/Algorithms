/**
 * @file MqttCommon.cpp
 * @brief Implementation of MQTT protocol helpers and MqttBase.
 */

#include "MqttCommon.hpp"
#include <cstring>

namespace Mqtt {

void SendAll(int sockFd, const std::vector<uint8_t>& buf) {
    size_t total = 0;
    while (total < buf.size()) {
        ssize_t sent = ::send(sockFd, buf.data() + total,
                              buf.size() - total, 0);
        if (sent <= 0) throw std::runtime_error("Failed to send data");
        total += static_cast<size_t>(sent);
    }
}

std::vector<uint8_t> RecvAll(int sockFd, size_t len) {
    std::vector<uint8_t> data(len);
    size_t total = 0;
    while (total < len) {
        ssize_t got = ::recv(sockFd, data.data() + total, len - total, 0);
        if (got <= 0) throw std::runtime_error("Failed to receive data");
        total += static_cast<size_t>(got);
    }
    return data;
}

std::vector<uint8_t> EncodeRemainingLength(size_t length) {
    std::vector<uint8_t> encoded;
    do {
        uint8_t byte = length % 128;
        length /= 128;
        if (length > 0) byte |= 0x80;
        encoded.push_back(byte);
    } while (length > 0);
    return encoded;
}

size_t DecodeRemainingLengthFromBuffer(const std::vector<uint8_t>& data,
                                       size_t offset,
                                       size_t& bytesConsumed) {
    size_t multiplier = 1;
    size_t value = 0;
    bytesConsumed = 0;
    while (offset < data.size()) {
        uint8_t encodedByte = data[offset++];
        ++bytesConsumed;
        value += (encodedByte & 0x7F) * multiplier;
        if ((encodedByte & 0x80) == 0) break;
        multiplier *= 128;
        if (multiplier > 128 * 128 * 128) {
            throw std::runtime_error("Malformed Remaining Length");
        }
    }
    return value;
}

size_t DecodeRemainingLength(int sockFd) {
    size_t multiplier = 1;
    size_t value = 0;
    while (true) {
        uint8_t encodedByte;
        if (::recv(sockFd, &encodedByte, 1, 0) != 1)
            throw std::runtime_error("Failed to read remaining length");
        value += (encodedByte & 0x7F) * multiplier;
        if ((encodedByte & 0x80) == 0) break;
        multiplier *= 128;
        if (multiplier > 128 * 128 * 128)
            throw std::runtime_error("Malformed Remaining Length");
    }
    return value;
}

std::vector<uint8_t> EncodeString(const std::string& s) {
    if (s.size() > UINT16_MAX) throw std::length_error("String too long");
    std::vector<uint8_t> out;
    out.push_back(static_cast<uint8_t>(s.size() >> 8));
    out.push_back(static_cast<uint8_t>(s.size() & 0xFF));
    out.insert(out.end(), s.begin(), s.end());
    return out;
}

// ---- MqttBase ----

MqttBase::MqttBase(const std::string& host, uint16_t port,
                   const std::string& clientId)
    : host_(host), port_(port), clientId_(clientId), sockFd_(-1) {}

MqttBase::~MqttBase() {
    if (sockFd_ != -1) ::close(sockFd_);
}

void MqttBase::ConnectToBroker(int keepAliveSec) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(),
                    &hints, &res) != 0)
        throw std::runtime_error("Address resolution failed");

    sockFd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockFd_ < 0) throw std::runtime_error("Socket creation failed");
    if (::connect(sockFd_, res->ai_addr, res->ai_addrlen) != 0)
        throw std::runtime_error("Connection to broker failed");
    freeaddrinfo(res);

    // Build CONNECT packet
    auto protoName = EncodeString("MQTT");
    std::vector<uint8_t> variableHeader;
    variableHeader.insert(variableHeader.end(), protoName.begin(), protoName.end());
    variableHeader.push_back(0x04); // Protocol Level 4
    variableHeader.push_back(0x02); // Clean Session
    variableHeader.push_back(static_cast<uint8_t>(keepAliveSec >> 8));
    variableHeader.push_back(static_cast<uint8_t>(keepAliveSec & 0xFF));

    auto payload = EncodeString(clientId_);
    size_t remainingLen = variableHeader.size() + payload.size();
    auto remLenEnc = EncodeRemainingLength(remainingLen);

    std::vector<uint8_t> packet;
    packet.push_back(0x10); // CONNECT
    packet.insert(packet.end(), remLenEnc.begin(), remLenEnc.end());
    packet.insert(packet.end(), variableHeader.begin(), variableHeader.end());
    packet.insert(packet.end(), payload.begin(), payload.end());

    SendAll(sockFd_, packet);

    auto resp = RecvAll(sockFd_, 4);
    if (resp[0] != 0x20 || resp[3] != 0x00)
        throw std::runtime_error("CONNACK error or unacceptable return code");
}

void MqttBase::Disconnect() {
    if (sockFd_ != -1) {
        SendAll(sockFd_, {0xE0, 0x00});
        ::close(sockFd_);
        sockFd_ = -1;
    }
}

} // namespace Mqtt
