/**
 * @file main.cpp
 * @brief Driver for the EpollServer module -- runs the echo server on port 8080.
 */

#include "EpollServer.hpp"
#include <iostream>

int main() {
    try {
        auto listenFd = Epoll::CreateListeningSocket(8080);
        auto epollFd  = Epoll::CreateEpollInstance(listenFd.Get());

        std::cout << "Server listening on port 8080\n";
        Epoll::RunEventLoop(epollFd.Get(), listenFd.Get());
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
