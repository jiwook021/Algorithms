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
#include <netdb.h>  // For getaddrinfo and gai_strerror

// Helper function to encode the remaining length for MQTT packets
std::vector<uint8_t> encodeRemainingLength(int length) {
    std::vector<uint8_t> encoded;
    do {
        uint8_t byte = length % 128; // Take the least significant 7 bits
        length /= 128;              // Shift right by 7 bits
        if (length > 0) {
            byte |= 128;            // Set continuation bit if more bytes follow
        }
        encoded.push_back(byte);
    } while (length > 0);
    return encoded;                 // Returns encoded bytes
}

// Namespace to organize MQTT-related classes
namespace mqtt {

    // Class representing an MQTT message
    class message {
    public:
        std::string topic;   // Topic to publish to
        std::string payload; // Message content
        int qos;             // Quality of Service (0, 1, or 2)
        bool retained;       // Whether the message should be retained by the broker

        // Constructor with validation for QoS
        message(const std::string& t, const std::string& p, int q = 0, bool r = false)
            : topic(t), payload(p), qos(q), retained(r) {
            if (qos < 0 || qos > 2) {
                throw std::invalid_argument("QoS must be 0, 1, or 2");
            }
        }
    };

    // Class to hold connection options for the MQTT client
    class connect_options {
    public:
        std::string brokerAddress;    // Broker URI (e.g., "tcp://localhost:1883")
        std::string clientId;         // Unique identifier for the client
        int keepAliveInterval;        // Keep-alive interval in seconds

        // Default constructor with default values
        connect_options()
            : brokerAddress("tcp://localhost:1883"),
              clientId("default_client"),
              keepAliveInterval(20) {}
    };

    // Class to manage the MQTT client connection and operations
    class async_client {
    private:
        std::string brokerHost;  // Hostname or IP of the broker
        int brokerPort;          // Port number of the broker
        std::string clientId;    // Client identifier
        int socketFd;            // File descriptor for the TCP socket

        // Parse the broker URI to extract host and port
        void parseBrokerAddress(const std::string& broker) {
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
                brokerHost = hostPort.substr(0, pos);
                brokerPort = std::stoi(hostPort.substr(pos + 1));
            } else {
                brokerHost = hostPort;
                brokerPort = 1883; // Default MQTT port
            }
        }

    public:
        // Constructor to initialize the client with broker and client ID
        async_client(const std::string& broker, const std::string& clientId)
            : brokerHost(""), brokerPort(0), clientId(clientId), socketFd(-1) {
            parseBrokerAddress(broker); // Set up host and port
        }

        // Destructor to close the socket if still open
        ~async_client() {
            if (socketFd >= 0) {
                close(socketFd); // Ensure socket is closed on destruction
            }
        }

        // Connect to the MQTT broker using the provided options
        void connect(const connect_options& opts) {
            // Create a TCP socket
            socketFd = socket(AF_INET, SOCK_STREAM, 0);
            if (socketFd < 0) {
                throw std::runtime_error("Failed to create socket");
            }

            // Set up the server address structure
            struct sockaddr_in serverAddr;
            memset(&serverAddr, 0, sizeof(serverAddr));
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(brokerPort);  // Convert port to network byte order

            // Resolve the hostname to an IP address using getaddrinfo
            struct addrinfo hints, *res;
            memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_INET;      // Use IPv4
            hints.ai_socktype = SOCK_STREAM; // TCP socket

            int status = getaddrinfo(brokerHost.c_str(), nullptr, &hints, &res);
            if (status != 0) {
                close(socketFd);
                throw std::runtime_error("Failed to resolve broker hostname: " + std::string(gai_strerror(status)));
            }

            // Copy the resolved IP address into serverAddr
            struct sockaddr_in* ip = (struct sockaddr_in*)res->ai_addr;
            serverAddr.sin_addr = ip->sin_addr;

            // Free the addrinfo structure
            freeaddrinfo(res);

            // Connect to the broker
            if (::connect(socketFd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
                close(socketFd);
                throw std::runtime_error("Connection failed");
            }

            // Build the MQTT CONNECT packet
            std::vector<uint8_t> packet;
            packet.push_back(0x10); // Fixed header: CONNECT type (0001 0000)

            // Variable header
            std::vector<uint8_t> variableHeader;
            // Protocol name: "MQTT" (with length prefix 0x0004)
            variableHeader.push_back(0x00);
            variableHeader.push_back(0x04);
            variableHeader.insert(variableHeader.end(), {'M', 'Q', 'T', 'T'});
            variableHeader.push_back(0x04); // Protocol level: 4 (MQTT 3.1.1)
            variableHeader.push_back(0x02); // Connect flags: clean session only
            // Keep-alive interval (16-bit)
            uint16_t keepAlive = opts.keepAliveInterval;
            variableHeader.push_back((keepAlive >> 8) & 0xFF); // High byte
            variableHeader.push_back(keepAlive & 0xFF);       // Low byte

            // Payload: client ID
            std::vector<uint8_t> payload;
            uint16_t idLen = opts.clientId.length();
            payload.push_back((idLen >> 8) & 0xFF); // Length high byte
            payload.push_back(idLen & 0xFF);        // Length low byte
            payload.insert(payload.end(), opts.clientId.begin(), opts.clientId.end());

            // Calculate and encode remaining length
            int remainingLength = variableHeader.size() + payload.size();
            std::vector<uint8_t> encodedRL = encodeRemainingLength(remainingLength);

            // Assemble the full packet
            packet.insert(packet.end(), encodedRL.begin(), encodedRL.end());
            packet.insert(packet.end(), variableHeader.begin(), variableHeader.end());
            packet.insert(packet.end(), payload.begin(), payload.end());

            // Send the CONNECT packet
            if (send(socketFd, packet.data(), packet.size(), 0) < 0) {
                close(socketFd);
                throw std::runtime_error("Failed to send CONNECT packet");
            }

            // Receive and check the CONNACK packet
            uint8_t buffer[4];
            if (recv(socketFd, buffer, 4, 0) < 4) {
                close(socketFd);
                throw std::runtime_error("Failed to receive CONNACK");
            }
            if (buffer[0] != 0x20 || buffer[1] != 0x02 || buffer[3] != 0x00) {
                close(socketFd);
                throw std::runtime_error("Connection refused by broker");
            }
        }

        // Publish a message to the specified topic
        void publish(const message& msg) {
            // Build the MQTT PUBLISH packet
            uint8_t fixedHeader = 0x30; // PUBLISH type (0011 0000), QoS 0, no retain
            if (msg.qos == 1) fixedHeader |= 0x02;       // Set QoS to 1
            else if (msg.qos == 2) fixedHeader |= 0x04;  // Set QoS to 2
            if (msg.retained) fixedHeader |= 0x01;       // Set retain flag

            // Variable header: topic name
            std::vector<uint8_t> variableHeader;
            uint16_t topicLen = msg.topic.length();
            variableHeader.push_back((topicLen >> 8) & 0xFF); // Length high byte
            variableHeader.push_back(topicLen & 0xFF);       // Length low byte
            variableHeader.insert(variableHeader.end(), msg.topic.begin(), msg.topic.end());

            // For QoS > 0, add a packet identifier (simplified with fixed ID)
            if (msg.qos > 0) {
                uint16_t packetId = 1; // Fixed packet ID (for simplicity)
                variableHeader.push_back((packetId >> 8) & 0xFF);
                variableHeader.push_back(packetId & 0xFF);
            }

            // Payload: message content
            std::vector<uint8_t> payload(msg.payload.begin(), msg.payload.end());

            // Calculate and encode remaining length
            int remainingLength = variableHeader.size() + payload.size();
            std::vector<uint8_t> encodedRL = encodeRemainingLength(remainingLength);

            // Assemble the full packet
            std::vector<uint8_t> packet;
            packet.push_back(fixedHeader);
            packet.insert(packet.end(), encodedRL.begin(), encodedRL.end());
            packet.insert(packet.end(), variableHeader.begin(), variableHeader.end());
            packet.insert(packet.end(), payload.begin(), payload.end());

            // Send the PUBLISH packet
            if (send(socketFd, packet.data(), packet.size(), 0) < 0) {
                throw std::runtime_error("Failed to send PUBLISH packet");
            }
            // Note: QoS 1 and 2 require acknowledgment handling, omitted here
        }

        // Disconnect from the broker
        void disconnect() {
            // Build and send the DISCONNECT packet
            uint8_t packet[2] = {0xE0, 0x00}; // DISCONNECT type (1110 0000), length 0
            send(socketFd, packet, 2, 0);
            close(socketFd);                  // Close the socket
            socketFd = -1;                    // Mark socket as closed
        }
    };

} // namespace mqtt

// Main function to demonstrate usage
int main() {
    std::string broker = "tcp://test.mosquitto.org:1883"; // Public test broker
    std::string clientId = "my_client";                   // Unique client ID

    mqtt::async_client client(broker, clientId);          // Create client instance
    mqtt::connect_options opts;                           // Connection options
    opts.brokerAddress = broker;
    opts.clientId = clientId;
    opts.keepAliveInterval = 20;

    try {
        std::cout << "Connecting to " << broker << "...\n";
        client.connect(opts);                             // Connect to broker
        std::cout << "Connected\n";

        mqtt::message msg("test/topic", "Hello, MQTT!", 0, false); // Create message
        std::cout << "Publishing to " << msg.topic << "...\n";
        client.publish(msg);                               // Publish message
        std::cout << "Published\n";

        std::cout << "Disconnecting...\n";
        client.disconnect();                              // Disconnect from broker
        std::cout << "Disconnected\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";       // Handle errors
        return 1;
    }
    return 0;
}