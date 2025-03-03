#include <iostream>
#include <vector>
#include "mqtt_common.h"

// Publisher class: only PUBLISH operation
class MqttPublisher : public MqttBase {
public:
    using MqttBase::MqttBase;

    // Publish a message with QoS0
    void Publish(const std::string& Topic, const std::string& Message) {
        auto TopicField = EncodeString(Topic);
        std::vector<uint8_t> Payload(Message.begin(), Message.end());

        size_t RemainingLen = TopicField.size() + Payload.size();
        auto RemLenEnc = EncodeRemainingLength(RemainingLen);

        std::vector<uint8_t> Packet;
        Packet.push_back(0x30); // PUBLISH, QoS0
        Packet.insert(Packet.end(), RemLenEnc.begin(), RemLenEnc.end());
        Packet.insert(Packet.end(), TopicField.begin(), TopicField.end());
        Packet.insert(Packet.end(), Payload.begin(), Payload.end());

        SendAll(Sockfd_, Packet);
    }
};

int main() {
    try {
        MqttPublisher Publisher("broker.hivemq.com", 1883, "cpp_pub_client");
        Publisher.ConnectToBroker();
        std::cout << "[+] Connected to broker" << std::endl;

        Publisher.Publish("test/topic", "Hello from C++ publisher!");
        std::cout << "[+] Message published" << std::endl;

        Publisher.Disconnect();
        std::cout << "[+] Disconnected" << std::endl;
    } catch (const std::exception& Ex) {
        std::cerr << "Error: " << Ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
