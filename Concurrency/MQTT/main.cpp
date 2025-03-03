/**
 * @file main.cpp
 * @brief Driver for the MQTT client simulation.
 *
 * Connects to a public MQTT test broker, publishes a message, and disconnects.
 *
 * Build:  make && ./main
 */

#include "Mqtt.hpp"

int main() {
    std::string broker   = "tcp://test.mosquitto.org:1883";
    std::string clientId = "my_client";

    Mqtt::AsyncClient client(broker, clientId);
    Mqtt::ConnectOptions opts;
    opts.BrokerAddress     = broker;
    opts.ClientId          = clientId;
    opts.KeepAliveInterval = 20;

    try {
        std::cout << "Connecting to " << broker << "...\n";
        client.Connect(opts);
        std::cout << "Connected\n";

        Mqtt::Message msg("test/topic", "Hello, MQTT!", 0, false);
        std::cout << "Publishing to " << msg.Topic << "...\n";
        client.Publish(msg);
        std::cout << "Published\n";

        std::cout << "Disconnecting...\n";
        client.Disconnect();
        std::cout << "Disconnected\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
