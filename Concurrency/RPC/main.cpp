/**
 * @file main.cpp
 * @brief Driver program for the RPC service demo.
 *
 * Starts an RPC server in a background thread, then uses an RPC client
 * to call the remote "add" function.
 *
 * Build:  make && ./main
 */

#include "RpcService.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

/**
 * @brief Sample function: adds two numbers.
 * @param args String arguments (expected: two integer strings).
 * @return Sum as a string, or error message.
 */
static std::string Add(const std::vector<std::string>& args) {
    if (args.size() != 2) {
        return "Error: Expected 2 arguments";
    }
    try {
        int a = std::stoi(args[0]);
        int b = std::stoi(args[1]);
        return std::to_string(a + b);
    } catch (const std::exception&) {
        return "Error: Invalid arguments";
    }
}

int main() {
    RpcServer server(8080);
    server.RegisterFunction("add", Add);

    std::thread serverThread([&server]() { server.Run(); });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    RpcClient client("127.0.0.1", 8080);
    std::vector<std::string> args = {"3", "5"};
    std::string result = client.Call("add", args);
    std::cout << "Result of add(3, 5): " << result << std::endl;

    serverThread.detach();
    return 0;
}
