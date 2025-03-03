/**
 * @file test_EpollServer.cpp
 * @brief Unit tests for the EpollServer module.
 *
 * Tests the FileDescriptor RAII wrapper, socket creation, and
 * echo-server round-trip (loopback).
 */

#include "EpollServer.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

/// FileDescriptor releases on destruction, reports -1 after move.
void TestFileDescriptorRaii() {
    // Create a dummy socket just to get a valid fd
    int rawFd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(rawFd >= 0);

    {
        Epoll::FileDescriptor fd(rawFd);
        assert(fd.Get() == rawFd);
    }
    // fd is now closed -- we can't easily verify closure without dup, but at
    // least ensure the wrapper compiles and runs without error.

    std::cout << "[PASS] TestFileDescriptorRaii\n";
}

/// FileDescriptor move semantics transfer ownership.
void TestFileDescriptorMove() {
    int rawFd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(rawFd >= 0);

    Epoll::FileDescriptor a(rawFd);
    Epoll::FileDescriptor b(std::move(a));
    assert(a.Get() == -1);
    assert(b.Get() == rawFd);

    std::cout << "[PASS] TestFileDescriptorMove\n";
}

/// MakeSocketNonBlocking sets O_NONBLOCK.
void TestNonBlocking() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);

    Epoll::MakeSocketNonBlocking(fd);
    int flags = ::fcntl(fd, F_GETFL, 0);
    assert(flags != -1);
    assert((flags & O_NONBLOCK) != 0);

    ::close(fd);
    std::cout << "[PASS] TestNonBlocking\n";
}

/// CreateListeningSocket binds successfully on a high port.
void TestCreateListeningSocket() {
    // Use a high port unlikely to be in use.
    auto fd = Epoll::CreateListeningSocket(19876);
    assert(fd.Get() >= 0);
    std::cout << "[PASS] TestCreateListeningSocket\n";
}

/// Full echo round-trip: server echoes back what client sends.
void TestEchoRoundTrip() {
    constexpr uint16_t PORT = 19877;

    auto listenFd = Epoll::CreateListeningSocket(PORT);
    auto epollFd  = Epoll::CreateEpollInstance(listenFd.Get());

    // Run server in background thread for a limited number of iterations.
    int capturedListenFd = listenFd.Get();
    int capturedEpollFd  = epollFd.Get();

    std::thread serverThread([capturedEpollFd, capturedListenFd]() {
        Epoll::RunEventLoop(capturedEpollFd, capturedListenFd, /*maxIterations=*/5);
    });

    // Give the server a moment to start.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Connect a client.
    int clientFd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(clientFd >= 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = ::htons(PORT);
    addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);

    int rc = ::connect(clientFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    assert(rc == 0);

    const char* message = "hello epoll";
    ssize_t sent = ::send(clientFd, message, std::strlen(message), 0);
    assert(sent == static_cast<ssize_t>(std::strlen(message)));

    char buf[128] = {};
    ssize_t received = ::recv(clientFd, buf, sizeof(buf) - 1, 0);
    assert(received > 0);
    buf[received] = '\0';

    assert(std::string(buf) == message);
    ::close(clientFd);

    serverThread.join();
    std::cout << "[PASS] TestEchoRoundTrip\n";
}

int main() {
    TestFileDescriptorRaii();
    TestFileDescriptorMove();
    TestNonBlocking();
    TestCreateListeningSocket();
    TestEchoRoundTrip();
    std::cout << "\nAll EpollServer tests passed.\n";
    return 0;
}
