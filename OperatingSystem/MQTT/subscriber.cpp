#include <iostream>
#include "mqtt_common.h"

// Subscriber class: SUBSCRIBE and receive messages
class MqttSubscriber : public MqttBase {
public:
    using MqttBase::MqttBase;

    // Subscribe to a topic with QoS0
    void Subscribe(const std::string& Topic) {
        uint16_t PacketId = 1;
        auto TopicField = EncodeString(Topic);
        std::vector<uint8_t> Payload;
        Payload.insert(Payload.end(), TopicField.begin(), TopicField.end());
        Payload.push_back(0x00); // QoS0

        std::vector<uint8_t> VariableHeader = {
            static_cast<uint8_t>(PacketId >> 8),
            static_cast<uint8_t>(PacketId & 0xFF)
        };

        size_t RemainingLen = VariableHeader.size() + Payload.size();
        auto RemLenEnc = EncodeRemainingLength(RemainingLen);

        std::vector<uint8_t> Packet;
        Packet.push_back(0x82); // SUBSCRIBE, QoS1 flag
        Packet.insert(Packet.end(), RemLenEnc.begin(), RemLenEnc.end());
        Packet.insert(Packet.end(), VariableHeader.begin(), VariableHeader.end());
        Packet.insert(Packet.end(), Payload.begin(), Payload.end());

        SendAll(Sockfd_, Packet);
        // Read SUBACK
        auto Header = RecvAll(Sockfd_, 2);
        size_t Len = Header[1];
        auto Body = RecvAll(Sockfd_, Len);
        if (Body[2] >= 0x80)
            throw std::runtime_error("SUBACK returned failure");
    }

    // Loop to receive incoming PUBLISH messages
    void ReceiveLoop() {
        while (true) {
            auto Header = RecvAll(Sockfd_, 1);
            uint8_t PacketType = Header[0] >> 4;
            if (PacketType != 3) continue; // Not PUBLISH

            size_t RemainingLen = DecodeRemainingLength(Sockfd_);
            auto Packet = RecvAll(Sockfd_, RemainingLen);

            uint16_t TopicLen = (Packet[0] << 8) | Packet[1];
            std::string Topic(Packet.begin() + 2, Packet.begin() + 2 + TopicLen);
            std::string Message(Packet.begin() + 2 + TopicLen, Packet.end());

            std::cout << "[Received] Topic: " << Topic
                      << " | Message: " << Message << std::endl;
        }
    }
};

int main() {
    try {
        MqttSubscriber Subscriber("broker.hivemq.com", 1883, "cpp_sub_client");
        Subscriber.ConnectToBroker();
        std::cout << "[+] Connected to broker" << std::endl;

        Subscriber.Subscribe("test/topic");
        std::cout << "[+] Subscribed to test/topic" << std::endl;

        Subscriber.ReceiveLoop(); // Blocks and prints incoming messages

        Subscriber.Disconnect();
    } catch (const std::exception& Ex) {
        std::cerr << "Error: " << Ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
