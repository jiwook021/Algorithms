# How do you use `epoll` or `select` in C++?

### Introduction to `epoll` and `select`

In network programming, `epoll` and `select` are mechanisms used to monitor multiple file descriptors to see if I/O is possible on any of them. They are mainly used in scenarios where a program needs to handle multiple simultaneous connections (e.g., in a web server). `select` is widely portable across various Unix-like systems, whereas `epoll` is specific to Linux but offers better performance in many scenarios, especially with a large number of file descriptors.

### Using `select` in C++

`select` allows you to monitor sets of file descriptors, waiting until one or more of the file descriptors become "ready" for some class of I/O operation (e.g., input readable).

Here is a simple example of how to use `select` in a C++ program:

```cpp
#include <iostream>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
    fd_set readfds;
    struct timeval tv;
    int retval;

    // Clear the set ahead of time
    FD_ZERO(&readfds);

    // Add our file descriptor to the set (0 is typically stdin)
    FD_SET(0, &readfds);

    // Wait up to five seconds.
    tv.tv_sec = 5;
    tv.tv_usec = 0;

    std::cout << "Waiting up to five seconds. Try typing something!" << std::endl;

    retval = select(1, &readfds, NULL, NULL, &tv);

    if (retval == -1) {
        perror("select()");
        return 1;
    } else if (retval) {
        std::cout << "Data is available now." << std::endl;
    } else {
        std::cout << "No data within five seconds." << std::endl;
    }

    return 0;
}
```

### Using `epoll` in C++

`epoll` is a Linux-specific I/O event notification facility, similar to `select` and `poll`, but with a capability to handle large numbers of file descriptors more efficiently.

Here is a basic example of using `epoll` in a C++ program:

```cpp
#include <sys/epoll.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

int main() {
    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        perror("epoll_create1");
        return 1;
    }

    struct epoll_event ev, events[10];
    ev.events = EPOLLIN;
    ev.data.fd = 0; // Monitoring stdin

    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, 0, &ev) == -1) {
        perror("epoll_ctl: stdin");
        close(epoll_fd);
        return 1;
    }

    std::cout << "Waiting for input. Try typing something!" << std::endl;

    while (true) {
        int nfds = epoll_wait(epoll_fd, events, 10, -1);
        if (nfds == -1) {
            perror("epoll_wait");
            close(epoll_fd);
            return 1;
        }

        for (int n = 0; n < nfds; ++n) {
            if (events[n].data.fd == 0) {
                char buffer[1024];
                ssize_t count = read(0, buffer, sizeof(buffer));
                if (count == -1) {
                    perror("read");
                    close(epoll_fd);
                    return 1;
                }
                std::cout << "Read: " << std::string(buffer, count) << std::endl;
            }
        }
    }

    close(epoll_fd);
    return 0;
}
```

### Summary

- **select**: Use if you need portability across different Unix-like systems and when you have a moderate number of file descriptors to monitor.
- **epoll**: Use for Linux-specific applications where you expect to handle a large number of active file descriptors simultaneously.

Both `select` and `epoll` can be used for building efficient event-driven applications, but `epoll` will generally perform better under high load, whereas `select` is easier to use and more portable.