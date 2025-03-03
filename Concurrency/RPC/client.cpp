#include <iostream>           
#include <string>             
#include <vector>            
#include <map>                
#include <functional>         // For function objects (e.g., std::function)
#include <sstream>            // For string stream operations (e.g., std::istringstream)
#include <sys/socket.h>       // For socket programming functions (e.g., socket, bind)
#include <netinet/in.h>       // For internet domain addresses (e.g., sockaddr_in)
#include <arpa/inet.h>        // For internet operations (e.g., inet_pton)
#include <unistd.h>           // For POSIX operating system API (e.g., close, read, write)
#include <thread>             // For multithreading support (e.g., std::thread)
#include <chrono>             // For time-related operations (e.g., std::chrono::seconds)

// RPC Server Class: Handles incoming client requests and executes registered functions
class RPCServer {
public:
    // Constructor: Initializes the server with a specified port
    RPCServer(int port) : port(port) {
        // Create a TCP socket using IPv4 (AF_INET) and stream type (SOCK_STREAM)
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) { // Check if socket creation failed
            std::cerr << "Socket creation failed\n";
            exit(1); // Exit the program with an error code
        }

        // Set up the server address structure
        sockaddr_in serverAddr;            // Structure to hold server address details
        serverAddr.sin_family = AF_INET;   // Use IPv4 address family
        serverAddr.sin_port = htons(port); // Convert port to network byte order (big-endian)
        serverAddr.sin_addr.s_addr = INADDR_ANY; // Accept connections on any network interface

        // Bind the socket to the specified address and port
        if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
            std::cerr << "Bind failed\n"; // Print error if binding fails
            close(serverSocket);          // Clean up the socket
            exit(1);                      // Exit the program
        }

        // Start listening for incoming connections with a backlog of 5 queued connections
        if (listen(serverSocket, 5) == -1) {
            std::cerr << "Listen failed\n"; // Print error if listening fails
            close(serverSocket);            // Clean up the socket
            exit(1);                        // Exit the program
        }

        std::cout << "Server listening on port " << port << "...\n"; // Indicate server is ready
    }

    // Destructor: Cleans up resources when the server object is destroyed
    ~RPCServer() {
        close(serverSocket); // Close the server socket to free up resources
    }

    void registerFunction(const std::string& name, std::function<std::string(const std::vector<std::string>&)> func) {
        functions[name] = func; // Store the function in the map with its name as the key
    }

    // Run the server: Accepts client connections and processes requests indefinitely
    void run() {
        while (true) { 
            sockaddr_in clientAddr;             // Structure to hold client address details
            socklen_t clientAddrSize = sizeof(clientAddr); // Size of the client address structure
            // Accept an incoming client connection, returning a new socket for communication
            int clientSocket = accept(serverSocket, (sockaddr*)&clientAddr, &clientAddrSize);
            if (clientSocket == -1) { // Check if accepting the connection failed
                std::cerr << "Accept failed\n";
                continue; // Skip to the next iteration of the loop
            }

            // Receive data from the client
            char buffer[1024] = {0}; // Buffer to store incoming data, initialized to zero
            int bytesReceived = read(clientSocket, buffer, sizeof(buffer) - 1); // Read data into buffer
            if (bytesReceived <= 0) { // Check if no data was received or an error occurred
                close(clientSocket);  // Close the client socket
                continue;             // Skip to the next iteration
            }

            // Parse the received request
            std::string request(buffer, bytesReceived); // Convert buffer to string
            std::istringstream iss(request);            // Create a string stream for parsing
            std::string functionName;                   // Variable to store the function name
            iss >> functionName;                        // Extract the function name from the request

            std::vector<std::string> args; // Vector to store function arguments
            std::string arg;               // Temporary variable for each argument
            while (iss >> arg) {           // Extract remaining arguments separated by spaces
                args.push_back(arg);       // Add each argument to the vector
            }

            // Execute the requested function if it exists
            std::string response; // Variable to store the response to send back
            if (functions.find(functionName) != functions.end()) { // Check if function is registered
                response = functions[functionName](args);          // Call the function with arguments
            } else {
                response = "Error: Function '" + functionName + "' not found"; // Error message if not found
            }

            // Send the response back to the client
            write(clientSocket, response.c_str(), response.size()); // Write response to client socket
            close(clientSocket);                                    // Close the client socket
        }
    }

private:
    int port;           // Port number the server listens on
    int serverSocket;   // File descriptor for the server socket
    // Map storing function names and their implementations
    std::map<std::string, std::function<std::string(const std::vector<std::string>&)>> functions;
};

// RPC Client Class: Connects to the server and calls remote functions
class RPCClient {
public:
    // Constructor: Initializes the client with the server's IP and port
    RPCClient(const std::string& serverIP, int serverPort) : serverIP(serverIP), serverPort(serverPort) {}

    // Call a remote function: Sends a request to the server and returns the result
    std::string call(const std::string& functionName, const std::vector<std::string>& args) {
        // Create a TCP socket for communication with the server
        int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (clientSocket == -1) { // Check if socket creation failed
            std::cerr << "Socket creation failed\n";
            return "Error: Socket creation failed"; // Return error message
        }

        // Set up the server address structure
        sockaddr_in serverAddr;                   // Structure to hold server address details
        serverAddr.sin_family = AF_INET;          // Use IPv4 address family
        serverAddr.sin_port = htons(serverPort);  // Convert port to network byte order
        // Convert IP address from string to binary form and store in serverAddr
        inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr);

        // Connect to the server
        if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
            std::cerr << "Connection failed\n"; // Print error if connection fails
            close(clientSocket);                // Clean up the socket
            return "Error: Connection failed";  // Return error message
        }

        // Construct the request string to send to the server
        std::string request = functionName; // Start with the function name
        for (const auto& arg : args) {      // Append each argument with a space separator
            request += " " + arg;
        }

        // Send the request to the server
        write(clientSocket, request.c_str(), request.size()); // Write the request to the socket

        // Receive the response from the server
        char buffer[1024] = {0}; // Buffer to store the response, initialized to zero
        int bytesReceived = read(clientSocket, buffer, sizeof(buffer) - 1); // Read response into buffer
        // Convert buffer to string if data was received, otherwise return an error
        std::string response = (bytesReceived > 0) ? std::string(buffer, bytesReceived) : "Error: No response";

        close(clientSocket); // Close the client socket
        return response;     // Return the response to the caller
    }

private:
    std::string serverIP; // IP address of the server
    int serverPort;       // Port number of the server
};

// Sample function: Adds two numbers provided as strings and returns the result as a string
std::string add(const std::vector<std::string>& args) {
    if (args.size() != 2) { // Check if exactly 2 arguments are provided
        return "Error: Expected 2 arguments"; // Return error if argument count is wrong
    }
    try {
        int a = std::stoi(args[0]); // Convert first argument to integer
        int b = std::stoi(args[1]); // Convert second argument to integer
        return std::to_string(a + b); // Add the numbers and return the result as a string
    } catch (const std::exception& e) { // Catch exceptions (e.g., invalid integer conversion)
        return "Error: Invalid arguments"; // Return error if conversion fails
    }
}

// Main function: Demonstrates the usage of RPCServer and RPCClient
int main() {
    // Create and start the server on port 8080
    RPCServer server(8080);               // Instantiate the server
    server.registerFunction("add", add);  // Register the 'add' function with the server
    // Run the server in a separate thread to allow concurrent client operations
    std::thread serverThread([&server]() { server.run(); });

    // Wait 1 second to ensure the server is fully started before the client connects
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Create a client and call the remote 'add' function
    RPCClient client("127.0.0.1", 8080);   // Instantiate the client connecting to localhost:8080
    std::vector<std::string> args = {"3", "5"}; // Arguments for the 'add' function
    std::string result = client.call("add", args); // Call the remote function and get the result
    std::cout << "Result of add(3, 5): " << result << std::endl; // Print the result

    // Clean up: Detach the server thread (note: in a real app, you'd implement proper shutdown)
    serverThread.detach(); // Allow the server thread to run independently
    return 0;              // Exit the program
}