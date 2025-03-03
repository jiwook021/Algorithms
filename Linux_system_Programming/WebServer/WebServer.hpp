#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <thread>
#include <vector>
#include <unordered_map>
#include <functional>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <atomic>
#include <chrono>

// Network headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

namespace WebServer {

// ============================================================================
// HTTP Request Parsing (testable without network)
// ============================================================================

struct HttpRequest {
    std::string method;
    std::string path;
    std::string httpVersion;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> queryParams;
    std::string body;
};

/// Parse a raw HTTP request string into an HttpRequest struct.
/// Returns true on success, false if the request line is malformed.
bool ParseHttpRequest(const std::string& rawData, HttpRequest& outRequest);

// ============================================================================
// HTTP Response Building (testable without network)
// ============================================================================

struct HttpResponse {
    int statusCode = 200;
    std::string statusText = "OK";
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    void SetStatus(int code, const std::string& text);
    void AddHeader(const std::string& name, const std::string& value);
    void SetBody(const std::string& content, const std::string& contentType = "text/html");

    /// Serialize to a raw HTTP response string.
    std::string Serialize() const;
};

// ============================================================================
// MIME Type Lookup (testable without network)
// ============================================================================

/// Return the MIME type for a given file extension (e.g. ".html" -> "text/html").
std::string GetMimeType(const std::string& extension);

// ============================================================================
// Route Handler
// ============================================================================

using RouteHandler = std::function<std::string(const std::unordered_map<std::string, std::string>&)>;

// ============================================================================
// Server
// ============================================================================

class Server {
public:
    Server(int port = 8080, const std::string& docRoot = "./www");
    ~Server();

    /// Register a route handler for a given path.
    void AddRoute(const std::string& path, RouteHandler handler);

    /// Start listening and accepting connections.
    bool Start();

    /// Stop the server.
    void Stop();

    /// Check whether the server is running.
    bool IsRunning() const;

private:
    void AcceptConnections();
    void HandleConnection(int clientSocket, const std::string& clientIp);
    std::string ProcessRequest(const std::string& rawRequest);

    int port_;
    std::string docRoot_;
    std::atomic<bool> running_;
    int serverSocket_ = -1;
    std::thread acceptThread_;

    std::mutex routeMutex_;
    std::unordered_map<std::string, RouteHandler> routes_;
};

} // namespace WebServer
