/**
 * @file main.cpp
 * @brief Driver for the WebSocketClient module.
 */

#include "WebSocketClient.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <host> <port> <message>\n";
        return EXIT_FAILURE;
    }

    const std::string host    = argv[1];
    const std::string port    = argv[2];
    const std::string message = argv[3];

    try {
        WebSocket::Client client;
        client.Connect(host, port);
        client.Send(message);

        std::string response = client.Receive();
        std::cout << "Received: " << response << "\n";

        client.Close();
    } catch (const boost::beast::system_error& e) {
        std::cerr << "[Beast Error] " << e.code().message() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "[Exception] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
