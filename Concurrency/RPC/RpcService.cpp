/**
 * @file RpcService.cpp
 * @brief Implementations for the TCP socket RPC server and client.
 */

#include "RpcService.hpp"

#include <arpa/inet.h>
#include <cstdlib>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace {

bool WriteAll(int socket, const std::string& message) {
    const char* data = message.c_str();
    std::size_t remaining = message.size();

    while (remaining > 0) {
        ssize_t bytesSent = write(socket, data, remaining);
        if (bytesSent <= 0) {
            return false;
        }

        data += bytesSent;
        remaining -= static_cast<std::size_t>(bytesSent);
    }

    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// RpcServer
// ---------------------------------------------------------------------------

RpcServer::RpcServer(int port) : port_(port) {
    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ == -1) {
        std::cerr << "Socket creation failed\n";
        std::exit(1);
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket_, reinterpret_cast<sockaddr*>(&serverAddr),
             sizeof(serverAddr)) == -1) {
        std::cerr << "Bind failed\n";
        close(serverSocket_);
        std::exit(1);
    }

    if (listen(serverSocket_, 5) == -1) {
        std::cerr << "Listen failed\n";
        close(serverSocket_);
        std::exit(1);
    }

    std::cout << "Server listening on port " << port_ << "...\n";
}

RpcServer::~RpcServer() {
    close(serverSocket_);
}

void RpcServer::RegisterFunction(
    const std::string& name,
    std::function<std::string(const std::vector<std::string>&)> func) {
    functions_[name] = func;
}

void RpcServer::Run() {
    while (true) {
        sockaddr_in clientAddr{};
        socklen_t clientAddrSize = sizeof(clientAddr);
        int clientSocket = accept(
            serverSocket_,
            reinterpret_cast<sockaddr*>(&clientAddr),
            &clientAddrSize);

        if (clientSocket == -1) {
            std::cerr << "Accept failed\n";
            continue;
        }

        char buffer[1024] = {0};
        int bytesReceived = read(clientSocket, buffer, sizeof(buffer) - 1);
        if (bytesReceived <= 0) {
            close(clientSocket);
            continue;
        }

        std::string request(buffer, bytesReceived);
        std::istringstream requestStream(request);
        std::string functionName;
        requestStream >> functionName;

        std::vector<std::string> args;
        std::string arg;
        while (requestStream >> arg) {
            args.push_back(arg);
        }

        std::string response;
        auto function = functions_.find(functionName);
        if (function != functions_.end()) {
            response = function->second(args);
        } else {
            response = "Error: Function '" + functionName + "' not found";
        }

        if (!WriteAll(clientSocket, response)) {
            std::cerr << "Response send failed\n";
        }
        close(clientSocket);
    }
}

// ---------------------------------------------------------------------------
// RpcClient
// ---------------------------------------------------------------------------

RpcClient::RpcClient(const std::string& serverIp, int serverPort)
    : serverIp_(serverIp), serverPort_(serverPort) {}

std::string RpcClient::Call(
    const std::string& functionName,
    const std::vector<std::string>& args) {
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Socket creation failed\n";
        return "Error: Socket creation failed";
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort_);
    inet_pton(AF_INET, serverIp_.c_str(), &serverAddr.sin_addr);

    if (connect(clientSocket, reinterpret_cast<sockaddr*>(&serverAddr),
                sizeof(serverAddr)) == -1) {
        std::cerr << "Connection failed\n";
        close(clientSocket);
        return "Error: Connection failed";
    }

    std::string request = functionName;
    for (const auto& arg : args) {
        request += " " + arg;
    }

    if (!WriteAll(clientSocket, request)) {
        close(clientSocket);
        return "Error: Request send failed";
    }

    char buffer[1024] = {0};
    int bytesReceived = read(clientSocket, buffer, sizeof(buffer) - 1);
    std::string response = (bytesReceived > 0)
        ? std::string(buffer, bytesReceived)
        : "Error: No response";

    close(clientSocket);
    return response;
}
