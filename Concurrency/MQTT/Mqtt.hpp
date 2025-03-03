/**
 * @file Mqtt.hpp
 * @brief Lightweight MQTT client/broker simulation.
 *
 * @details
 * Simulates an MQTT publish-subscribe client using raw TCP sockets.
 * Supports CONNECT, PUBLISH, and DISCONNECT packet construction
 * per the MQTT 3.1.1 specification. Includes topic subscriptions,
 * QoS levels (0, 1, 2), and retain flags.
 */

#ifndef MQTT_HPP
#define MQTT_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <netdb.h>

/**
 * @brief Encode the MQTT remaining-length field.
 * @param length The length value to encode.
 * @return Vector of encoded bytes.
 */
inline std::vector<uint8_t> EncodeRemainingLength(int length) {
    std::vector<uint8_t> encoded;
    do {
        uint8_t byte = length % 128;
        length /= 128;
        if (length > 0) {
            byte |= 128;
        }
        encoded.push_back(byte);
    } while (length > 0);
    return encoded;
}

/** @brief Namespace for all MQTT-related classes. */
namespace Mqtt {

/**
 * @class Message
 * @brief Represents an MQTT message with topic, payload, QoS, and retain flag.
 */
class Message {
public:
    std::string Topic;   ///< Topic to publish to.
    std::string Payload; ///< Message content.
    int Qos;             ///< Quality of Service (0, 1, or 2).
    bool Retained;       ///< Whether the message should be retained.

    /**
     * @brief Construct a Message.
     * @param topic   Topic string.
     * @param payload Payload string.
     * @param qos     QoS level (0-2).
     * @param retained Retain flag.
     * @throws std::invalid_argument if QoS is not 0, 1, or 2.
     */
    Message(const std::string& topic, const std::string& payload,
            int qos = 0, bool retained = false)
        : Topic(topic), Payload(payload), Qos(qos), Retained(retained) {
        if (Qos < 0 || Qos > 2) {
            throw std::invalid_argument("QoS must be 0, 1, or 2");
        }
    }
};

/**
 * @class ConnectOptions
 * @brief Holds connection options for the MQTT client.
 */
class ConnectOptions {
public:
    std::string BrokerAddress;   ///< Broker URI (e.g., "tcp://localhost:1883").
    std::string ClientId;        ///< Unique identifier for the client.
    int KeepAliveInterval;       ///< Keep-alive interval in seconds.

    /** @brief Default constructor with sensible defaults. */
    ConnectOptions()
        : BrokerAddress("tcp://localhost:1883"),
          ClientId("default_client"),
          KeepAliveInterval(20) {}
};

/**
 * @class AsyncClient
 * @brief Manages an MQTT client connection over TCP sockets.
 *
 * Supports connecting to an MQTT broker, publishing messages, and
 * disconnecting. Constructs raw MQTT 3.1.1 packets.
 */
class AsyncClient {
private:
    std::string brokerHost_;  ///< Hostname or IP of the broker.
    int brokerPort_;          ///< Port number of the broker.
    std::string clientId_;    ///< Client identifier.
    int socketFd_;            ///< File descriptor for the TCP socket.

    /**
     * @brief Parse the broker URI to extract host and port.
     * @param broker URI string (e.g., "tcp://host:port").
     * @throws std::runtime_error if format is invalid or protocol is not TCP.
     */
    void ParseBrokerAddress(const std::string& broker) {
        size_t pos = broker.find("://");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid broker address format");
        }
        std::string protocol = broker.substr(0, pos);
        if (protocol != "tcp") {
            throw std::runtime_error("Only TCP protocol is supported");
        }
        std::string hostPort = broker.substr(pos + 3);
        pos = hostPort.find(":");
        if (pos != std::string::npos) {
            brokerHost_ = hostPort.substr(0, pos);
            brokerPort_ = std::stoi(hostPort.substr(pos + 1));
        } else {
            brokerHost_ = hostPort;
            brokerPort_ = 1883;
        }
    }

public:
    /**
     * @brief Construct an AsyncClient.
     * @param broker   Broker URI (e.g., "tcp://localhost:1883").
     * @param clientId Unique client identifier.
     */
    AsyncClient(const std::string& broker, const std::string& clientId)
        : brokerHost_(""), brokerPort_(0), clientId_(clientId), socketFd_(-1) {
        ParseBrokerAddress(broker);
    }

    /** @brief Destructor closes the socket if still open. */
    ~AsyncClient() {
        if (socketFd_ >= 0) {
            close(socketFd_);
        }
    }

    /** @brief Non-copyable. */
    AsyncClient(const AsyncClient&) = delete;
    /** @brief Non-copyable. */
    AsyncClient& operator=(const AsyncClient&) = delete;

    /**
     * @brief Connect to the MQTT broker.
     * @param opts Connection options.
     * @throws std::runtime_error on connection failure.
     */
    void Connect(const ConnectOptions& opts) {
        socketFd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socketFd_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(brokerPort_);

        struct addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        int status = getaddrinfo(brokerHost_.c_str(), nullptr, &hints, &res);
        if (status != 0) {
            close(socketFd_);
            throw std::runtime_error(
                "Failed to resolve broker hostname: " +
                std::string(gai_strerror(status)));
        }

        auto* ip = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
        serverAddr.sin_addr = ip->sin_addr;
        freeaddrinfo(res);

        if (::connect(socketFd_, reinterpret_cast<struct sockaddr*>(&serverAddr),
                      sizeof(serverAddr)) < 0) {
            close(socketFd_);
            throw std::runtime_error("Connection failed");
        }

        // Build CONNECT packet
        std::vector<uint8_t> packet;
        packet.push_back(0x10);

        std::vector<uint8_t> variableHeader;
        variableHeader.push_back(0x00);
        variableHeader.push_back(0x04);
        variableHeader.insert(variableHeader.end(), {'M', 'Q', 'T', 'T'});
        variableHeader.push_back(0x04);
        variableHeader.push_back(0x02);
        uint16_t keepAlive = opts.KeepAliveInterval;
        variableHeader.push_back((keepAlive >> 8) & 0xFF);
        variableHeader.push_back(keepAlive & 0xFF);

        std::vector<uint8_t> payload;
        uint16_t idLen = opts.ClientId.length();
        payload.push_back((idLen >> 8) & 0xFF);
        payload.push_back(idLen & 0xFF);
        payload.insert(payload.end(), opts.ClientId.begin(), opts.ClientId.end());

        int remainingLength = variableHeader.size() + payload.size();
        auto encodedRl = EncodeRemainingLength(remainingLength);

        packet.insert(packet.end(), encodedRl.begin(), encodedRl.end());
        packet.insert(packet.end(), variableHeader.begin(), variableHeader.end());
        packet.insert(packet.end(), payload.begin(), payload.end());

        if (send(socketFd_, packet.data(), packet.size(), 0) < 0) {
            close(socketFd_);
            throw std::runtime_error("Failed to send CONNECT packet");
        }

        uint8_t buffer[4];
        if (recv(socketFd_, buffer, 4, 0) < 4) {
            close(socketFd_);
            throw std::runtime_error("Failed to receive CONNACK");
        }
        if (buffer[0] != 0x20 || buffer[1] != 0x02 || buffer[3] != 0x00) {
            close(socketFd_);
            throw std::runtime_error("Connection refused by broker");
        }
    }

    /**
     * @brief Publish a message to the broker.
     * @param msg The Message to publish.
     * @throws std::runtime_error on send failure.
     */
    void Publish(const Message& msg) {
        uint8_t fixedHeader = 0x30;
        if (msg.Qos == 1) fixedHeader |= 0x02;
        else if (msg.Qos == 2) fixedHeader |= 0x04;
        if (msg.Retained) fixedHeader |= 0x01;

        std::vector<uint8_t> variableHeader;
        uint16_t topicLen = msg.Topic.length();
        variableHeader.push_back((topicLen >> 8) & 0xFF);
        variableHeader.push_back(topicLen & 0xFF);
        variableHeader.insert(variableHeader.end(),
                              msg.Topic.begin(), msg.Topic.end());

        if (msg.Qos > 0) {
            uint16_t packetId = 1;
            variableHeader.push_back((packetId >> 8) & 0xFF);
            variableHeader.push_back(packetId & 0xFF);
        }

        std::vector<uint8_t> payload(msg.Payload.begin(), msg.Payload.end());

        int remainingLength = variableHeader.size() + payload.size();
        auto encodedRl = EncodeRemainingLength(remainingLength);

        std::vector<uint8_t> packet;
        packet.push_back(fixedHeader);
        packet.insert(packet.end(), encodedRl.begin(), encodedRl.end());
        packet.insert(packet.end(), variableHeader.begin(), variableHeader.end());
        packet.insert(packet.end(), payload.begin(), payload.end());

        if (send(socketFd_, packet.data(), packet.size(), 0) < 0) {
            throw std::runtime_error("Failed to send PUBLISH packet");
        }
    }

    /**
     * @brief Disconnect from the broker.
     */
    void Disconnect() {
        uint8_t packet[2] = {0xE0, 0x00};
        send(socketFd_, packet, 2, 0);
        close(socketFd_);
        socketFd_ = -1;
    }
};

} // namespace Mqtt

#endif // MQTT_HPP
