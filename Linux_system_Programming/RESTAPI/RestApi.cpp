#include "RestApi.hpp"

namespace RestApi {

// ============================================================================
// Logger Implementation
// ============================================================================

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : level_(Level::INFO), fileLogging_(false) {}

void Logger::SetLevel(Level level) { level_ = level; }

bool Logger::SetLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    fileStream_.close();
    fileStream_.open(filename, std::ios::out | std::ios::app);
    if (!fileStream_.is_open()) {
        std::cerr << "Cannot open log file: " << filename << std::endl;
        return false;
    }
    fileLogging_ = true;
    return true;
}

void Logger::Log(Level level, const std::string& message) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    std::stringstream ss;
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    ss << " [" << LevelToString(level) << "] " << message;

    std::string logEntry = ss.str();
    std::cout << logEntry << std::endl;

    if (fileLogging_ && fileStream_.is_open()) {
        fileStream_ << logEntry << std::endl;
        fileStream_.flush();
    }
}

void Logger::Debug(const std::string& message)   { Log(Level::DEBUG, message); }
void Logger::Info(const std::string& message)    { Log(Level::INFO, message); }
void Logger::Warning(const std::string& message) { Log(Level::WARNING, message); }
void Logger::Error(const std::string& message)   { Log(Level::ERROR, message); }
void Logger::Fatal(const std::string& message)   { Log(Level::FATAL, message); }

std::string Logger::LevelToString(Level level) {
    switch (level) {
        case Level::DEBUG:   return "DEBUG  ";
        case Level::INFO:    return "INFO   ";
        case Level::WARNING: return "WARNING";
        case Level::ERROR:   return "ERROR  ";
        case Level::FATAL:   return "FATAL  ";
        default:             return "UNKNOWN";
    }
}

// ============================================================================
// HTTP Helpers
// ============================================================================

std::string HttpMethodToString(HttpMethod method) {
    switch (method) {
        case HttpMethod::GET:    return "GET";
        case HttpMethod::POST:   return "POST";
        case HttpMethod::PUT:    return "PUT";
        case HttpMethod::DELETE: return "DELETE";
        case HttpMethod::PATCH:  return "PATCH";
        default:                 return "UNKNOWN";
    }
}

std::string HttpStatusToString(HttpStatus status) {
    switch (status) {
        case HttpStatus::OK:                    return "200 OK";
        case HttpStatus::CREATED:               return "201 Created";
        case HttpStatus::ACCEPTED:              return "202 Accepted";
        case HttpStatus::NO_CONTENT:            return "204 No Content";
        case HttpStatus::BAD_REQUEST:           return "400 Bad Request";
        case HttpStatus::UNAUTHORIZED:          return "401 Unauthorized";
        case HttpStatus::FORBIDDEN:             return "403 Forbidden";
        case HttpStatus::NOT_FOUND:             return "404 Not Found";
        case HttpStatus::METHOD_NOT_ALLOWED:    return "405 Method Not Allowed";
        case HttpStatus::INTERNAL_SERVER_ERROR: return "500 Internal Server Error";
        case HttpStatus::NOT_IMPLEMENTED:       return "501 Not Implemented";
        default:                                return "Unknown Status";
    }
}

std::string HttpRequest::ToString() const {
    std::stringstream ss;
    ss << HttpMethodToString(method) << " " << path;
    if (!queryParams.empty()) {
        ss << "?";
        bool first = true;
        for (const auto& [key, value] : queryParams) {
            if (!first) ss << "&";
            first = false;
            ss << key << "=" << value;
        }
    }
    return ss.str();
}

std::string HttpResponse::ToString() const {
    return HttpStatusToString(status) + ", Body length: " + std::to_string(body.size());
}

// ============================================================================
// Json Implementation
// ============================================================================

Json::JsonObject Json::Parse(const std::string& jsonStr) {
    JsonObject result;
    size_t pos = jsonStr.find('{');
    size_t endPos = jsonStr.rfind('}');

    if (pos == std::string::npos || endPos == std::string::npos) {
        LOG_WARNING("Invalid JSON format: missing braces");
        return result;
    }

    std::string content = jsonStr.substr(pos + 1, endPos - pos - 1);
    size_t start = 0;
    size_t end;

    while ((end = content.find(',', start)) != std::string::npos || start < content.length()) {
        size_t nextStart;
        if (end != std::string::npos) {
            nextStart = end + 1;
        } else {
            end = content.length();
            nextStart = end;
        }

        std::string pair = content.substr(start, end - start);
        size_t colon = pair.find(':');

        if (colon != std::string::npos) {
            std::string key = Trim(pair.substr(0, colon));
            std::string value = Trim(pair.substr(colon + 1));

            if (key.front() == '"' && key.back() == '"')
                key = key.substr(1, key.length() - 2);

            if (value.front() == '"' && value.back() == '"') {
                result[key] = value.substr(1, value.length() - 2);
            } else if (value == "true") {
                result[key] = true;
            } else if (value == "false") {
                result[key] = false;
            } else if (value == "null") {
                result[key] = nullptr;
            } else {
                try {
                    size_t idx;
                    if (value.find('.') != std::string::npos) {
                        double dVal = std::stod(value, &idx);
                        if (idx == value.length()) result[key] = dVal;
                    } else {
                        int iVal = std::stoi(value, &idx);
                        if (idx == value.length()) result[key] = iVal;
                    }
                } catch (...) {
                    result[key] = value;
                }
            }
        }

        start = nextStart;
        if (end == content.length()) break;
    }

    return result;
}

std::string Json::Stringify(const JsonObject& obj) {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto& [key, value] : obj) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << key << "\":";
        std::visit([&ss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>)
                ss << "\"" << arg << "\"";
            else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, double>)
                ss << arg;
            else if constexpr (std::is_same_v<T, bool>)
                ss << (arg ? "true" : "false");
            else if constexpr (std::is_same_v<T, std::nullptr_t>)
                ss << "null";
        }, value);
    }
    ss << "}";
    return ss.str();
}

std::string Json::Trim(const std::string& str) {
    const auto begin = std::find_if_not(str.begin(), str.end(),
                                         [](char c) { return std::isspace(c); });
    const auto end = std::find_if_not(str.rbegin(), str.rend(),
                                       [](char c) { return std::isspace(c); }).base();
    return (begin < end) ? std::string(begin, end) : std::string();
}

// ============================================================================
// Router Implementation
// ============================================================================

bool Router::RouteKey::operator==(const RouteKey& other) const {
    return method == other.method && path == other.path;
}

std::size_t Router::RouteKeyHash::operator()(const RouteKey& key) const {
    return std::hash<std::string>()(std::to_string(static_cast<int>(key.method)) + key.path);
}

void Router::RegisterHandler(HttpMethod method, const std::string& path, HandlerFunc handler) {
    routes_[{method, path}] = std::move(handler);
    LOG_INFO("Route registered: " + HttpMethodToString(method) + " " + path);
}

HttpResponse Router::Route(const HttpRequest& request) {
    auto it = routes_.find({request.method, request.path});
    if (it != routes_.end()) {
        try {
            return it->second(request);
        } catch (const std::exception& e) {
            LOG_ERROR("Handler exception: " + std::string(e.what()));
            return {HttpStatus::INTERNAL_SERVER_ERROR,
                    {{"Content-Type", "application/json"}},
                    "{\"error\": \"Internal server error: " + std::string(e.what()) + "\"}"};
        }
    }
    LOG_WARNING("Route not found: " + request.ToString());
    return {HttpStatus::NOT_FOUND,
            {{"Content-Type", "application/json"}},
            "{\"error\": \"Not found\"}"};
}

void Router::PrintRoutes() const {
    LOG_INFO("Registered routes:");
    for (const auto& [routeKey, _] : routes_) {
        LOG_INFO("  " + HttpMethodToString(routeKey.method) + " " + routeKey.path);
    }
}

// ============================================================================
// HttpParser Implementation
// ============================================================================

std::optional<HttpRequest> HttpParser::Parse(const std::string& rawData) {
    HttpRequest request;
    std::istringstream stream(rawData);
    std::string line;

    if (!std::getline(stream, line) || !ParseRequestLine(line, request))
        return std::nullopt;

    while (std::getline(stream, line) && !line.empty() && line != "\r") {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;
        ParseHeader(line, request.headers);
    }

    if (request.headers.contains("Content-Length")) {
        size_t contentLength = 0;
        try {
            contentLength = std::stoul(request.headers["Content-Length"]);
        } catch (...) {
            return std::nullopt;
        }
        if (contentLength > 0) {
            std::string body;
            char c;
            while (stream.get(c) && body.size() < contentLength)
                body.push_back(c);
            request.body = body;
        }
    }
    return request;
}

bool HttpParser::ParseRequestLine(const std::string& line, HttpRequest& request) {
    std::istringstream lineStream(line);
    std::string methodStr, pathWithQuery, version;
    if (!(lineStream >> methodStr >> pathWithQuery >> version)) return false;

    if (methodStr == "GET")         request.method = HttpMethod::GET;
    else if (methodStr == "POST")   request.method = HttpMethod::POST;
    else if (methodStr == "PUT")    request.method = HttpMethod::PUT;
    else if (methodStr == "DELETE") request.method = HttpMethod::DELETE;
    else if (methodStr == "PATCH")  request.method = HttpMethod::PATCH;
    else                            request.method = HttpMethod::UNKNOWN;

    auto queryPos = pathWithQuery.find('?');
    if (queryPos != std::string::npos) {
        request.path = pathWithQuery.substr(0, queryPos);
        ParseQueryParams(pathWithQuery.substr(queryPos + 1), request.queryParams);
    } else {
        request.path = pathWithQuery;
    }
    return true;
}

void HttpParser::ParseHeader(const std::string& line,
                              std::unordered_map<std::string, std::string>& headers) {
    auto pos = line.find(':');
    if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        headers[key] = value;
    }
}

void HttpParser::ParseQueryParams(const std::string& queryString,
                                   std::unordered_map<std::string, std::string>& params) {
    std::istringstream stream(queryString);
    std::string pair;
    while (std::getline(stream, pair, '&')) {
        auto eqPos = pair.find('=');
        if (eqPos != std::string::npos)
            params[pair.substr(0, eqPos)] = pair.substr(eqPos + 1);
        else
            params[pair] = "";
    }
}

// ============================================================================
// HttpFormatter Implementation
// ============================================================================

std::string HttpFormatter::Format(const HttpResponse& response) {
    std::stringstream ss;
    ss << "HTTP/1.1 " << static_cast<int>(response.status) << " "
       << GetStatusText(response.status) << "\r\n";
    for (const auto& [name, value] : response.headers)
        ss << name << ": " << value << "\r\n";
    if (!response.headers.contains("Content-Length") && !response.body.empty())
        ss << "Content-Length: " << response.body.size() << "\r\n";
    ss << "\r\n";
    if (!response.body.empty()) ss << response.body;
    return ss.str();
}

std::string HttpFormatter::GetStatusText(HttpStatus status) {
    switch (status) {
        case HttpStatus::OK:                    return "OK";
        case HttpStatus::CREATED:               return "Created";
        case HttpStatus::ACCEPTED:              return "Accepted";
        case HttpStatus::NO_CONTENT:            return "No Content";
        case HttpStatus::BAD_REQUEST:           return "Bad Request";
        case HttpStatus::UNAUTHORIZED:          return "Unauthorized";
        case HttpStatus::FORBIDDEN:             return "Forbidden";
        case HttpStatus::NOT_FOUND:             return "Not Found";
        case HttpStatus::METHOD_NOT_ALLOWED:    return "Method Not Allowed";
        case HttpStatus::INTERNAL_SERVER_ERROR: return "Internal Server Error";
        case HttpStatus::NOT_IMPLEMENTED:       return "Not Implemented";
        default:                                return "Unknown";
    }
}

// ============================================================================
// WebServer Implementation
// ============================================================================

WebServer::WebServer(uint16_t port, size_t threadPoolSize)
    : port_(port), threadPoolSize_(threadPoolSize), running_(false) {
    signal(SIGPIPE, SIG_IGN);
    LOG_INFO("Web server initialized. Port: " + std::to_string(port) +
             ", Thread pool size: " + std::to_string(threadPoolSize));
}

WebServer::~WebServer() { Stop(); }

void WebServer::RegisterHandler(HttpMethod method, const std::string& path,
                                 Router::HandlerFunc handler) {
    router_.RegisterHandler(method, path, std::move(handler));
}

bool WebServer::Start() {
    LOG_INFO("Server start attempt...");

    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        LOG_ERROR("Socket creation failed: " + std::string(std::strerror(errno)));
        return false;
    }

    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        LOG_ERROR("Socket option setting failed: " + std::string(std::strerror(errno)));
        close(serverSocket_);
        return false;
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(serverSocket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        LOG_ERROR("Binding failed: " + std::string(std::strerror(errno)));
        close(serverSocket_);
        return false;
    }

    if (listen(serverSocket_, 10) < 0) {
        LOG_ERROR("Listen failed: " + std::string(std::strerror(errno)));
        close(serverSocket_);
        return false;
    }

    running_ = true;

    for (size_t i = 0; i < threadPoolSize_; ++i) {
        workerThreads_.emplace_back([this]() { WorkerLoop(); });
    }

    acceptorThread_ = std::thread([this]() { AcceptorLoop(); });

    LOG_INFO("Server started on port " + std::to_string(port_));
    router_.PrintRoutes();
    return true;
}

void WebServer::Stop() {
    if (!running_) return;
    LOG_INFO("Stopping server...");
    running_ = false;

    if (serverSocket_ >= 0) {
        close(serverSocket_);
        serverSocket_ = -1;
    }

    if (acceptorThread_.joinable()) acceptorThread_.join();

    cv_.notify_all();
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) thread.join();
    }
    workerThreads_.clear();

    {
        std::lock_guard<std::mutex> lock(clientsMutex_);
        for (int clientSocket : clientSockets_) close(clientSocket);
        clientSockets_.clear();
    }
    LOG_INFO("Server stopped successfully.");
}

void WebServer::AcceptorLoop() {
    while (running_) {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket_, (struct sockaddr*)&clientAddr, &clientLen);

        if (clientSocket < 0) {
            if (running_)
                LOG_ERROR("Client accept failed: " + std::string(std::strerror(errno)));
            continue;
        }

        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, INET_ADDRSTRLEN);
        LOG_INFO("New client: " + std::string(clientIp) + ":" +
                 std::to_string(ntohs(clientAddr.sin_port)));

        int flags = fcntl(clientSocket, F_GETFL, 0);
        fcntl(clientSocket, F_SETFL, flags | O_NONBLOCK);

        {
            std::lock_guard<std::mutex> lock(clientsMutex_);
            clientSockets_.push_back(clientSocket);
        }
        cv_.notify_one();
    }
}

void WebServer::WorkerLoop() {
    while (running_) {
        int clientSocket = -1;
        {
            std::unique_lock<std::mutex> lock(clientsMutex_);
            cv_.wait(lock, [this]() { return !running_ || !clientSockets_.empty(); });
            if (!running_ && clientSockets_.empty()) return;
            if (!clientSockets_.empty()) {
                clientSocket = clientSockets_.back();
                clientSockets_.pop_back();
            }
        }
        if (clientSocket >= 0) {
            ProcessClient(clientSocket);
            close(clientSocket);
        }
    }
}

void WebServer::ProcessClient(int clientSocket) {
    std::string requestData;
    char buffer[4096];

    struct pollfd fd;
    fd.fd = clientSocket;
    fd.events = POLLIN;

    int pollResult = poll(&fd, 1, 5000);
    if (pollResult > 0 && (fd.revents & POLLIN)) {
        ssize_t bytesRead;
        while ((bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytesRead] = '\0';
            requestData.append(buffer, bytesRead);

            if (requestData.find("\r\n\r\n") != std::string::npos) {
                size_t headerEnd = requestData.find("\r\n\r\n") + 4;
                size_t contentLength = 0;
                size_t clPos = requestData.find("Content-Length:");
                if (clPos != std::string::npos) {
                    size_t clEnd = requestData.find("\r\n", clPos);
                    std::string clValue = requestData.substr(clPos + 16, clEnd - clPos - 16);
                    try { contentLength = std::stoul(clValue); } catch (...) {}
                }
                if (requestData.size() >= headerEnd + contentLength) break;
            }

            pollResult = poll(&fd, 1, 1000);
            if (pollResult <= 0 || !(fd.revents & POLLIN)) break;
        }

        if (!requestData.empty()) HandleRequest(clientSocket, requestData);
    }
}

void WebServer::HandleRequest(int clientSocket, const std::string& requestData) {
    auto requestOpt = HttpParser::Parse(requestData);

    if (!requestOpt) {
        HttpResponse response = {HttpStatus::BAD_REQUEST,
                                  {{"Content-Type", "application/json"}},
                                  "{\"error\": \"Bad request\"}"};
        std::string responseStr = HttpFormatter::Format(response);
        send(clientSocket, responseStr.c_str(), responseStr.size(), 0);
        return;
    }

    const HttpRequest& request = *requestOpt;
    LOG_INFO("Request: " + request.ToString());
    HttpResponse response = router_.Route(request);
    std::string responseStr = HttpFormatter::Format(response);
    send(clientSocket, responseStr.c_str(), responseStr.size(), 0);
}

} // namespace RestApi
