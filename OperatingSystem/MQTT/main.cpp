/**
 * @file main.cpp
 * @brief Driver demonstrating MQTT publish (requires a running MQTT broker).
 */

#include "MqttCommon.hpp"
#include <iostream>

/// Simple publisher using MqttBase.
class MqttPublisher : public Mqtt::MqttBase {
public:
    using MqttBase::MqttBase;

    void Publish(const std::string& topic, const std::string& message) {
        auto topicField = Mqtt::EncodeString(topic);
        std::vector<uint8_t> payload(message.begin(), message.end());

        size_t remainingLen = topicField.size() + payload.size();
        auto remLenEnc = Mqtt::EncodeRemainingLength(remainingLen);

        std::vector<uint8_t> packet;
        packet.push_back(0x30); // PUBLISH, QoS0
        packet.insert(packet.end(), remLenEnc.begin(), remLenEnc.end());
        packet.insert(packet.end(), topicField.begin(), topicField.end());
        packet.insert(packet.end(), payload.begin(), payload.end());

        Mqtt::SendAll(sockFd_, packet);
    }
};

int main() {
    try {
        MqttPublisher publisher("broker.hivemq.com", 1883, "cpp_pub_client");
        publisher.ConnectToBroker();
        std::cout << "[+] Connected to broker\n";

        publisher.Publish("test/topic", "Hello from C++ publisher!");
        std::cout << "[+] Message published\n";

        publisher.Disconnect();
        std::cout << "[+] Disconnected\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
