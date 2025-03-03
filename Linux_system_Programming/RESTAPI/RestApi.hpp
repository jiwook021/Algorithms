#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <fstream>
#include <variant>
#include <optional>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <atomic>

// Linux/POSIX headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <arpa/inet.h>
#include <signal.h>
#include <cerrno>
#include <cstring>

namespace RestApi {

// ============================================================================
// Logger
// ============================================================================

class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERROR, FATAL };

    static Logger& GetInstance();
    void SetLevel(Level level);
    bool SetLogFile(const std::string& filename);
    void Log(Level level, const std::string& message);
    void Debug(const std::string& message);
    void Info(const std::string& message);
    void Warning(const std::string& message);
    void Error(const std::string& message);
    void Fatal(const std::string& message);

private:
    Logger();
    std::string LevelToString(Level level);

    Level level_;
    std::mutex mutex_;
    bool fileLogging_;
    std::ofstream fileStream_;
};

#define LOG_DEBUG(msg)   RestApi::Logger::GetInstance().Debug(msg)
#define LOG_INFO(msg)    RestApi::Logger::GetInstance().Info(msg)
#define LOG_WARNING(msg) RestApi::Logger::GetInstance().Warning(msg)
#define LOG_ERROR(msg)   RestApi::Logger::GetInstance().Error(msg)
#define LOG_FATAL(msg)   RestApi::Logger::GetInstance().Fatal(msg)

// ============================================================================
// HTTP Types
// ============================================================================

enum class HttpStatus {
    OK = 200,
    CREATED = 201,
    ACCEPTED = 202,
    NO_CONTENT = 204,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    INTERNAL_SERVER_ERROR = 500,
    NOT_IMPLEMENTED = 501
};

enum class HttpMethod { GET, POST, PUT, DELETE, PATCH, UNKNOWN };

std::string HttpMethodToString(HttpMethod method);
std::string HttpStatusToString(HttpStatus status);

struct HttpRequest {
    HttpMethod method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> queryParams;
    std::string body;

    std::string ToString() const;
};

struct HttpResponse {
    HttpStatus status;
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    std::string ToString() const;
};

// ============================================================================
// Json
// ============================================================================

class Json {
public:
    using JsonObject = std::unordered_map<std::string, std::variant<std::string, int, double, bool, std::nullptr_t>>;

    static JsonObject Parse(const std::string& jsonStr);
    static std::string Stringify(const JsonObject& obj);

private:
    static std::string Trim(const std::string& str);
};

// ============================================================================
// Router
// ============================================================================

class Router {
public:
    using HandlerFunc = std::function<HttpResponse(const HttpRequest&)>;

    void RegisterHandler(HttpMethod method, const std::string& path, HandlerFunc handler);
    HttpResponse Route(const HttpRequest& request);
    void PrintRoutes() const;

private:
    struct RouteKey {
        HttpMethod method;
        std::string path;
        bool operator==(const RouteKey& other) const;
    };

    struct RouteKeyHash {
        std::size_t operator()(const RouteKey& key) const;
    };

    std::unordered_map<RouteKey, HandlerFunc, RouteKeyHash> routes_;
};

// ============================================================================
// HttpParser
// ============================================================================

class HttpParser {
public:
    static std::optional<HttpRequest> Parse(const std::string& rawData);

private:
    static bool ParseRequestLine(const std::string& line, HttpRequest& request);
    static void ParseHeader(const std::string& line,
                            std::unordered_map<std::string, std::string>& headers);
    static void ParseQueryParams(const std::string& queryString,
                                 std::unordered_map<std::string, std::string>& params);
};

// ============================================================================
// HttpFormatter
// ============================================================================

class HttpFormatter {
public:
    static std::string Format(const HttpResponse& response);

private:
    static std::string GetStatusText(HttpStatus status);
};

// ============================================================================
// WebServer
// ============================================================================

class WebServer {
public:
    WebServer(uint16_t port, size_t threadPoolSize = 4);
    ~WebServer();

    void RegisterHandler(HttpMethod method, const std::string& path,
                         Router::HandlerFunc handler);
    bool Start();
    void Stop();

private:
    void AcceptorLoop();
    void WorkerLoop();
    void ProcessClient(int clientSocket);
    void HandleRequest(int clientSocket, const std::string& requestData);

    uint16_t port_;
    size_t threadPoolSize_;
    std::atomic<bool> running_;

    int serverSocket_ = -1;
    std::vector<int> clientSockets_;
    std::mutex clientsMutex_;
    std::condition_variable cv_;

    std::thread acceptorThread_;
    std::vector<std::thread> workerThreads_;

    Router router_;
};

} // namespace RestApi
