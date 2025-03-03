/**
 * @file EpollServer.hpp
 * @brief Epoll-based TCP echo server declarations
 * @details RAII file descriptor wrapper, non-blocking socket helpers,
 *          and a simple epoll echo server class.
 */

#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Epoll {

/// RAII wrapper for a POSIX file descriptor.
class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) noexcept;
    ~FileDescriptor();

    FileDescriptor(FileDescriptor&& other) noexcept;
    FileDescriptor& operator=(FileDescriptor&& other) noexcept;

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int Get() const noexcept;
    void Reset(int newFd);

private:
    int fd_;
};

/// Set a socket to non-blocking mode.
void MakeSocketNonBlocking(int fd);

/// Create, bind, and listen on a TCP socket.
FileDescriptor CreateListeningSocket(uint16_t port);

/// Create an epoll instance and register a file descriptor.
FileDescriptor CreateEpollInstance(int listenFd);

/// Run the echo-server event loop (blocking).
/// @param maxIterations  if > 0, stop after that many epoll_wait rounds (for testing).
void RunEventLoop(int epollFd, int listenFd, int maxIterations = 0);

} // namespace Epoll
