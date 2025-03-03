// client.cpp
// A simple TCP client in C++23.
// Connects to localhost:8080, sends a line from stdin, then prints the server echo.

#include <arpa/inet.h>      // inet_pton, htons
#include <netinet/in.h>     // sockaddr_in
#include <sys/socket.h>     // socket, connect
#include <unistd.h>         // close, read, write

#include <cerrno>           // errno
#include <cstring>          // strerror
#include <exception>        // std::exception
#include <iostream>         // std::cout, std::cerr
#include <string>           // std::string

int main() {
    try {
        // === 1) Create the socket ===
        // AF_INET = IPv4, SOCK_STREAM = TCP
        int SockFd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (SockFd < 0) {
            throw std::runtime_error("socket() failed: " + std::string(std::strerror(errno)));
        }

        // === 2) Server address setup ===
        sockaddr_in ServerAddr{};
        ServerAddr.sin_family = AF_INET;            // IPv4
        ServerAddr.sin_port = ::htons(8080);        // port 8080
        // Convert dotted-decimal IP to in_addr
        if (::inet_pton(AF_INET,
                        "127.0.0.1",
                        &ServerAddr.sin_addr) <= 0)
        {
            throw std::runtime_error("inet_pton() failed: invalid IP address");
        }

        // === 3) Connect to the server ===
        if (::connect(SockFd,
                      reinterpret_cast<sockaddr*>(&ServerAddr),
                      sizeof(ServerAddr)) < 0)
        {
            throw std::runtime_error("connect() failed: " + std::string(std::strerror(errno)));
        }
        std::cout << "Connected to 127.0.0.1:8080\n";

        // === 4) Read user input from stdin ===
        std::cout << "Enter a line to send: ";
        std::string InputLine;
        if (!std::getline(std::cin, InputLine)) {
            std::cerr << "No input detected. Exiting.\n";
            ::close(SockFd);
            return EXIT_FAILURE;
        }
        // Append newline so server sees end-of-line
        InputLine.push_back('\n');

        // === 5) Send the entire inputLine ===
        ssize_t TotalSent = 0;
        ssize_t ToSend    = static_cast<ssize_t>(InputLine.size());
        const char* data  = InputLine.data();

        while (TotalSent < ToSend) {
            ssize_t Sent = ::send(SockFd,
                                  data + TotalSent,
                                  ToSend - TotalSent,
                                  0);
            if (Sent < 0) {
                throw std::runtime_error("send() failed: " + std::string(std::strerror(errno)));
            }
            TotalSent += Sent;  // track how much was sent
        }

        // === 6) Receive echo back from server ===
        constexpr size_t BUFFER_SIZE = 4096;
        char Buffer[BUFFER_SIZE];

        ssize_t Received = ::recv(SockFd, Buffer, BUFFER_SIZE - 1, 0);
        if (Received < 0) {
            throw std::runtime_error("recv() failed: " + std::string(std::strerror(errno)));
        }
        if (Received == 0) {
            std::cerr << "Server closed connection unexpectedly.\n";
        } else {
            Buffer[Received] = '\0';  // null-terminate
            std::cout << "Received echo: " << Buffer;
        }

        // === 7) Clean up and exit ===
        ::close(SockFd);
    }
    catch (const std::exception& Ex) {
        // Print any errors and exit non-zero
        std::cerr << "Error: " << Ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
