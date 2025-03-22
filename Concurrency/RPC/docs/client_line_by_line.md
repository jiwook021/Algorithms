# Step-by-Step Explanation: client.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in detail, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the design choices.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>           
#include <string>             
#include <vector>            
#include <map>                
#include <functional>         
#include <sstream>            
#include <sys/socket.h>       
#include <netinet/in.h>       
#include <arpa/inet.h>        
#include <unistd.h>           
#include <thread>             
#include <chrono>             
```

#### **What It Does**
These lines include libraries that provide functionality for:
- Input/output (`<iostream>`).
- String manipulation (`<string>`).
- Dynamic arrays (`<vector>`).
- Key-value pairs (`<map>`).
- Function objects (`<functional>`).
- String parsing (`<sstream>`).
- Socket programming (`<sys/socket.h>`, `<netinet/in.h>`, `<arpa/inet.h>`).
- Operating system APIs (`<unistd.h>`).
- Multithreading (`<thread>`).
- Time-related operations (`<chrono>`).

#### **Why These Are Used**
- **`<iostream>`**: For printing messages to the console (e.g., `std::cout`).
- **`<string>`**: For handling text data (e.g., function names and arguments).
- **`<vector>`**: For storing lists of arguments (e.g., `{"3", "5"}`).
- **`<map>`**: For storing registered functions (e.g., `"add"` maps to the `add` function).
- **`<functional>`**: For storing and calling functions dynamically.
- **`<sstream>`**: For parsing strings (e.g., converting `"3"` to the number `3`).
- **Socket libraries**: For network communication between the client and server.
- **`<thread>`**: For running the server in a separate thread.
- **`<chrono>`**: For adding delays (e.g., waiting 1 second for the server to start).

---

### **2. RPCServer Class**
```cpp
class RPCServer {
public:
    RPCServer(int port) : port(port) {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) {
            std::cerr << "Socket creation failed\n";
            exit(1);
        }

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        serverAddr.sin_addr.s_addr = INADDR_ANY;

        if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
            std::cerr << "Bind failed\n";
            close(serverSocket);
            exit(1);
        }
    }
};
```

#### **What It Does**
This is the constructor for the `RPCServer` class. It:
1. Creates a TCP socket.
2. Sets up the server address.
3. Binds the socket to the specified port.

#### **Step-by-Step Breakdown**
1. **Socket Creation**:
   - `socket(AF_INET, SOCK_STREAM, 0)` creates a TCP socket.
     - `AF_INET`: Specifies IPv4 addressing.
     - `SOCK_STREAM`: Specifies a TCP socket (reliable, connection-oriented).
     - `0`: Default protocol (TCP).
   - If the socket creation fails (`serverSocket == -1`), the program exits with an error.

2. **Server Address Setup**:
   - `sockaddr_in serverAddr`: A structure to hold the server’s address details.
     - `sin_family`: Set to `AF_INET` (IPv4).
     - `sin_port`: The port number, converted to network byte order using `htons()`.
     - `sin_addr.s_addr`: Set to `INADDR_ANY`, meaning the server accepts connections on all network interfaces.

3. **Binding the Socket**:
   - `bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr))` binds the socket to the address and port.
   - If binding fails, the program prints an error, closes the socket, and exits.

#### **Why These Steps Are Necessary**
- **Socket Creation**: Establishes a communication endpoint.
- **Address Setup**: Specifies where the server will listen for connections.
- **Binding**: Associates the socket with the address and port.

---

### **3. Function Registration**
```cpp
void registerFunction(const std::string& name, std::function<int(int, int)> func) {
    functions[name] = func;
}
```

#### **What It Does**
This method registers a function (e.g., `add`) with the server. The function is stored in a `std::map` with its name as the key.

#### **Step-by-Step Breakdown**
1. **Parameters**:
   - `name`: The name of the function (e.g., `"add"`).
   - `func`: The function itself (e.g., a lambda or function pointer).

2. **Storage**:
   - `functions[name] = func`: Stores the function in a `std::map` called `functions`.

#### **Why This Is Used**
- **`std::map`**: Allows quick lookup of functions by name.
- **`std::function`**: Enables storing and calling functions dynamically.

---

### **4. Server Execution**
```cpp
void run() {
    listen(serverSocket, 5);
    while (true) {
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        std::thread([this, clientSocket]() {
            handleClient(clientSocket);
        }).detach();
    }
}
```

#### **What It Does**
This method:
1. Listens for incoming connections.
2. Accepts a connection from a client.
3. Handles the client in a separate thread.

#### **Step-by-Step Breakdown**
1. **Listening**:
   - `listen(serverSocket, 5)`: Marks the socket as a passive socket that will accept incoming connections. The `5` specifies the maximum number of pending connections.

2. **Accepting Connections**:
   - `accept(serverSocket, nullptr, nullptr)`: Blocks until a client connects, then returns a new socket for communication with the client.

3. **Handling Clients**:
   - A new thread is created to handle the client using `std::thread`.
   - `detach()` allows the thread to run independently.

#### **Why This Is Used**
- **Multithreading**: Allows the server to handle multiple clients concurrently.
- **`detach()`**: Ensures the server doesn’t wait for the client thread to finish.

---

### **5. Main Function**
```cpp
int main() {
    RPCServer server(8080);
    server.registerFunction("add", [](int a, int b) { return a + b; });
    std::thread serverThread([&server]() { server.run(); });
    std::this_thread::sleep_for(std::chrono::seconds(1));

    RPCClient client("127.0.0.1", 8080);
    std::vector<std::string> args = {"3", "5"};
    std::string result = client.call("add", args);
    std::cout << "Result of add(3, 5): " << result << std::endl;
}
```

#### **What It Does**
1. Creates and starts the server.
2. Registers the `add` function.
3. Runs the server in a separate thread.
4. Waits for the server to start.
5. Creates a client and calls the `add` function remotely.

#### **Step-by-Step Breakdown**
1. **Server Initialization**:
   - `RPCServer server(8080)`: Creates a server listening on port 8080.
   - `server.registerFunction("add", [](int a, int b) { return a + b; })`: Registers the `add` function.

2. **Server Thread**:
   - `std::thread serverThread([&server]() { server.run(); })`: Starts the server in a separate thread.

3. **Client Call**:
   - `RPCClient client("127.0.0.1", 8080)`: Creates a client connecting to `localhost:8080`.
   - `client.call("add", args)`: Calls the `add` function with arguments `3` and `5`.

4. **Result Handling**:
   - The result is printed to the console.

#### **Why This Is Used**
- **Threading**: Ensures the server can handle clients while the main thread continues.
- **Delays**: Ensures the server is ready before the client connects.

---

### **6. Diagram of Control Flow**
```
+-------------------+       +-------------------+       +-------------------+
| Server Initialization | --> | Function Registration | --> | Server Execution |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
        v                           v                           v
+-------------------+       +-------------------+       +-------------------+
| Client Initialization | --> | Remote Function Call | --> | Result Handling |
+-------------------+       +-------------------+       +-------------------+
```

---

This explanation should make the code accessible to everyone, from beginners to experts. Let me know if you’d like further clarification or additional examples!