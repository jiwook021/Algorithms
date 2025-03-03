/**
 * @file EpollServer.cpp
 * @brief Implementation of the epoll echo server.
 */

#include "EpollServer.hpp"

namespace Epoll {

// ---- FileDescriptor ----

FileDescriptor::FileDescriptor(int fd) noexcept : fd_(fd) {}

FileDescriptor::~FileDescriptor() {
    if (fd_ != -1) {
        ::close(fd_);
    }
}

FileDescriptor::FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
}

FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
        if (fd_ != -1) ::close(fd_);
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

int FileDescriptor::Get() const noexcept { return fd_; }

void FileDescriptor::Reset(int newFd) {
    if (fd_ != -1) ::close(fd_);
    fd_ = newFd;
}

// ---- Helpers ----

void MakeSocketNonBlocking(int fd) {
    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        throw std::runtime_error("fcntl(F_GETFL) failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        throw std::runtime_error("fcntl(F_SETFL) failed: " +
                                 std::string(std::strerror(errno)));
    }
}

FileDescriptor CreateListeningSocket(uint16_t port) {
    FileDescriptor listenFd{::socket(AF_INET, SOCK_STREAM, 0)};
    if (listenFd.Get() < 0) {
        throw std::runtime_error("socket() failed: " +
                                 std::string(std::strerror(errno)));
    }

    int yes = 1;
    if (::setsockopt(listenFd.Get(), SOL_SOCKET, SO_REUSEADDR, &yes,
                     sizeof(yes)) < 0) {
        throw std::runtime_error("setsockopt(SO_REUSEADDR) failed: " +
                                 std::string(std::strerror(errno)));
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = ::htons(port);

    if (::bind(listenFd.Get(), reinterpret_cast<sockaddr*>(&serverAddr),
               sizeof(serverAddr)) < 0) {
        throw std::runtime_error("bind() failed: " +
                                 std::string(std::strerror(errno)));
    }

    if (::listen(listenFd.Get(), SOMAXCONN) < 0) {
        throw std::runtime_error("listen() failed: " +
                                 std::string(std::strerror(errno)));
    }

    MakeSocketNonBlocking(listenFd.Get());
    return listenFd;
}

FileDescriptor CreateEpollInstance(int listenFd) {
    FileDescriptor epollFd{::epoll_create1(0)};
    if (epollFd.Get() < 0) {
        throw std::runtime_error("epoll_create1() failed: " +
                                 std::string(std::strerror(errno)));
    }

    epoll_event event{};
    event.events = EPOLLIN | EPOLLET;
    event.data.fd = listenFd;

    if (::epoll_ctl(epollFd.Get(), EPOLL_CTL_ADD, listenFd, &event) < 0) {
        throw std::runtime_error("epoll_ctl ADD listenFd failed: " +
                                 std::string(std::strerror(errno)));
    }

    return epollFd;
}

void RunEventLoop(int epollFd, int listenFd, int maxIterations) {
    std::vector<epoll_event> events(64);
    int iteration = 0;

    while (maxIterations <= 0 || iteration < maxIterations) {
        int timeout = (maxIterations > 0) ? 100 : -1; // short timeout for tests
        int nReady = ::epoll_wait(epollFd, events.data(),
                                  static_cast<int>(events.size()), timeout);
        if (nReady < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("epoll_wait() failed: " +
                                     std::string(std::strerror(errno)));
        }

        for (int i = 0; i < nReady; ++i) {
            int readyFd = events[i].data.fd;

            if (readyFd == listenFd) {
                // Accept new connections
                while (true) {
                    sockaddr_in clientAddr{};
                    socklen_t addrLen = sizeof(clientAddr);
                    int clientFd = ::accept(listenFd,
                                            reinterpret_cast<sockaddr*>(&clientAddr),
                                            &addrLen);
                    if (clientFd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        std::cerr << "accept() error: " << std::strerror(errno) << "\n";
                        break;
                    }
                    MakeSocketNonBlocking(clientFd);

                    epoll_event clientEvent{};
                    clientEvent.events = EPOLLIN | EPOLLET;
                    clientEvent.data.fd = clientFd;
                    if (::epoll_ctl(epollFd, EPOLL_CTL_ADD, clientFd,
                                    &clientEvent) < 0) {
                        std::cerr << "epoll_ctl ADD clientFd failed: "
                                  << std::strerror(errno) << "\n";
                        ::close(clientFd);
                    } else {
                        char ipStr[INET_ADDRSTRLEN];
                        ::inet_ntop(AF_INET, &clientAddr.sin_addr, ipStr,
                                    sizeof(ipStr));
                        std::cout << "Accepted connection from " << ipStr << ":"
                                  << ntohs(clientAddr.sin_port) << "\n";
                    }
                }
            } else if (events[i].events & EPOLLIN) {
                // Read and echo
                constexpr size_t BUF_SIZE = 4096;
                char buffer[BUF_SIZE];

                while (true) {
                    ssize_t count = ::read(readyFd, buffer, BUF_SIZE);
                    if (count < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        std::cerr << "read() error: " << std::strerror(errno) << "\n";
                        ::close(readyFd);
                        break;
                    }
                    if (count == 0) {
                        std::cout << "Client disconnected (fd=" << readyFd << ")\n";
                        ::close(readyFd);
                        break;
                    }

                    ssize_t sentTotal = 0;
                    while (sentTotal < count) {
                        ssize_t sent = ::write(readyFd, buffer + sentTotal,
                                               count - sentTotal);
                        if (sent < 0) {
                            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                            std::cerr << "write() error: " << std::strerror(errno) << "\n";
                            ::close(readyFd);
                            break;
                        }
                        sentTotal += sent;
                    }
                }
            } else {
                std::cerr << "Unexpected event on fd " << readyFd
                          << ": events=" << events[i].events << "\n";
                ::close(readyFd);
            }
        }

        ++iteration;
    }
}

} // namespace Epoll
