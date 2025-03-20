# Implement a simple TCP client/server in C++.

Implementing a simple TCP client/server application in C++ involves using the socket programming API provided by the operating system. Below, I will provide a basic example of a TCP server and a TCP client using C++. The server will wait for a connection from the client, receive a message, and then send a response back. The client will connect to the server, send a message, and wait for a response.

We will use the POSIX sockets API, which is available on Unix-like systems such as Linux and macOS. For Windows, the socket API is slightly different (it requires some additional initialization), but for simplicity, this example will focus on POSIX.

### 1. TCP Server

Create a file named `tcp_server.cpp`:

```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    const int port = 8080;
    
    char buffer[1024] = {0};
    const char* hello = "Hello from server";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Bind the socket to the address
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // Accept a connection
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // Read the message from the client
    read(new_socket, buffer, 1024);
    std::cout << "Message from client: " << buffer << std::endl;

    // Send a response to the client
    send(new_socket, hello, strlen(hello), 0);
    std::cout << "Hello message sent\n";
    
    // Close the socket
    close(new_socket);
    close(server_fd);

    return 0;
}
```

### 2. TCP Client

Create a file named `tcp_client.cpp`:

```cpp
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    const char* hello = "Hello from client";
    char buffer[1024] = {0};

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "\n Socket creation error \n";
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cout << "\nInvalid address/ Address not supported \n";
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cout << "\nConnection Failed \n";
        return -1;
    }

    send(sock, hello, strlen(hello), 0);
    std::cout << "Hello message sent\n";
    int valread = read(sock, buffer, 1024);
    std::cout << "Server: " << buffer << std::endl;

    // Close the socket
    close(sock);

    return 0;
}
```

### How to Compile and Run

1. Open a terminal.
2. Navigate to the directory containing the files.
3. Compile the server and client using `g++`:
   ```bash
   g++ tcp_server.cpp -o tcp_server
   g++ tcp_client.cpp -o tcp_client
   ```
4. First, run the server:
   ```bash
   ./tcp_server
   ```
5. Open another terminal and run the client:
   ```bash
   ./tcp_client
   ```

This simple example demonstrates basic TCP communication between a client and a server in C++. Each side sends and receives a single message. For real applications, you would need to implement error handling, support multiple clients, and potentially use threads or asynchronous I/O to handle I/O operations efficiently.