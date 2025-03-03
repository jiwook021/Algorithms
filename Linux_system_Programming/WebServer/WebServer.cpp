#include "WebServer.hpp"

namespace WebServer {

// ============================================================================
// HTTP Request Parsing
// ============================================================================

bool ParseHttpRequest(const std::string& rawData, HttpRequest& outRequest) {
    std::istringstream requestStream(rawData);
    std::string line;

    // Parse request line (e.g. "GET /index.html HTTP/1.1")
    if (!std::getline(requestStream, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::istringstream lineStream(line);
    if (!(lineStream >> outRequest.method >> outRequest.path >> outRequest.httpVersion))
        return false;

    // Separate query parameters from path
    size_t queryPos = outRequest.path.find('?');
    if (queryPos != std::string::npos) {
        std::string queryString = outRequest.path.substr(queryPos + 1);
        outRequest.path = outRequest.path.substr(0, queryPos);

        std::istringstream queryStream(queryString);
        std::string param;
        while (std::getline(queryStream, param, '&')) {
            size_t eqPos = param.find('=');
            if (eqPos != std::string::npos)
                outRequest.queryParams[param.substr(0, eqPos)] = param.substr(eqPos + 1);
            else
                outRequest.queryParams[param] = "";
        }
    }

    // Parse headers
    while (std::getline(requestStream, line) && !line.empty()) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;

        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string key = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            outRequest.headers[key] = value;
        }
    }

    return true;
}

// ============================================================================
// HTTP Response
// ============================================================================

void HttpResponse::SetStatus(int code, const std::string& text) {
    statusCode = code;
    statusText = text;
}

void HttpResponse::AddHeader(const std::string& name, const std::string& value) {
    headers[name] = value;
}

void HttpResponse::SetBody(const std::string& content, const std::string& contentType) {
    body = content;
    headers["Content-Type"] = contentType;
    headers["Content-Length"] = std::to_string(body.size());
}

std::string HttpResponse::Serialize() const {
    std::stringstream ss;
    ss << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";
    for (const auto& [key, value] : headers)
        ss << key << ": " << value << "\r\n";
    if (headers.find("Connection") == headers.end())
        ss << "Connection: close\r\n";
    ss << "\r\n" << body;
    return ss.str();
}

// ============================================================================
// MIME Type Lookup
// ============================================================================

std::string GetMimeType(const std::string& extension) {
    if (extension == ".html" || extension == ".htm") return "text/html";
    if (extension == ".css")  return "text/css";
    if (extension == ".js")   return "application/javascript";
    if (extension == ".json") return "application/json";
    if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
    if (extension == ".png")  return "image/png";
    if (extension == ".gif")  return "image/gif";
    if (extension == ".svg")  return "image/svg+xml";
    if (extension == ".txt")  return "text/plain";
    if (extension == ".xml")  return "application/xml";
    if (extension == ".pdf")  return "application/pdf";
    return "application/octet-stream";
}

// ============================================================================
// Server Implementation
// ============================================================================

Server::Server(int port, const std::string& docRoot)
    : port_(port), docRoot_(docRoot), running_(false) {
    if (!std::filesystem::exists(docRoot_)) {
        std::filesystem::create_directories(docRoot_);
        std::ofstream indexFile(docRoot_ + "/index.html");
        indexFile << "<html><head><title>C++ Web Server</title></head>"
                  << "<body><h1>Welcome to C++ Web Server</h1></body></html>";
        indexFile.close();
    }
}

Server::~Server() { Stop(); }

void Server::AddRoute(const std::string& path, RouteHandler handler) {
    std::lock_guard<std::mutex> lock(routeMutex_);
    routes_[path] = handler;
}

bool Server::Start() {
    if (running_) {
        std::cerr << "Server is already running!" << std::endl;
        return false;
    }

    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ == -1) {
        std::cerr << "Failed to create socket: " << strerror(errno) << std::endl;
        return false;
    }

    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Failed to set socket options: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port_);

    if (bind(serverSocket_, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Failed to bind socket: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }

    if (listen(serverSocket_, 10) < 0) {
        std::cerr << "Failed to listen: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }

    running_ = true;
    std::cout << "Server started on port " << port_ << std::endl;
    std::cout << "Document root: " << docRoot_ << std::endl;

    acceptThread_ = std::thread(&Server::AcceptConnections, this);
    return true;
}

void Server::Stop() {
    if (!running_) return;
    running_ = false;

    if (serverSocket_ != -1) {
        close(serverSocket_);
        serverSocket_ = -1;
    }

    if (acceptThread_.joinable()) acceptThread_.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Server stopped" << std::endl;
}

bool Server::IsRunning() const { return running_; }

void Server::AcceptConnections() {
    while (running_) {
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket_, (struct sockaddr*)&clientAddr, &clientAddrLen);
        if (clientSocket < 0) {
            if (running_)
                std::cerr << "Failed to accept connection: " << strerror(errno) << std::endl;
            continue;
        }

        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIp, INET_ADDRSTRLEN);

        int flags = fcntl(clientSocket, F_GETFL, 0);
        fcntl(clientSocket, F_SETFL, flags | O_NONBLOCK);

        std::thread worker(&Server::HandleConnection, this, clientSocket, std::string(clientIp));
        worker.detach();
    }
}

void Server::HandleConnection(int clientSocket, const std::string& /* clientIp */) {
    constexpr size_t BUFFER_SIZE = 8192;
    std::vector<char> buffer(BUFFER_SIZE);
    std::string request;

    auto startTime = std::chrono::steady_clock::now();
    constexpr auto TIMEOUT = std::chrono::seconds(10);

    while (true) {
        if (std::chrono::steady_clock::now() - startTime > TIMEOUT) break;

        ssize_t bytesRead = recv(clientSocket, buffer.data(), buffer.size() - 1, 0);
        if (bytesRead > 0) {
            buffer[bytesRead] = '\0';
            request.append(buffer.data(), bytesRead);
            if (request.find("\r\n\r\n") != std::string::npos) break;
        } else if (bytesRead == 0) {
            break;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            break;
        }
    }

    if (!request.empty()) {
        std::string response = ProcessRequest(request);
        size_t totalSent = 0;
        while (totalSent < response.size()) {
            ssize_t sent = send(clientSocket, response.c_str() + totalSent,
                                response.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                break;
            } else if (sent == 0) {
                break;
            }
            totalSent += sent;
        }
    }

    close(clientSocket);
}

std::string Server::ProcessRequest(const std::string& rawRequest) {
    HttpRequest req;
    if (!ParseHttpRequest(rawRequest, req)) {
        HttpResponse resp;
        resp.SetStatus(400, "Bad Request");
        resp.SetBody("<html><body><h1>400 Bad Request</h1></body></html>");
        return resp.Serialize();
    }

    std::string path = req.path;
    if (path == "/") path = "/index.html";

    // Check route handlers
    {
        std::lock_guard<std::mutex> lock(routeMutex_);
        auto it = routes_.find(path);
        if (it != routes_.end()) {
            std::string content = it->second(req.queryParams);
            HttpResponse resp;
            resp.SetStatus(200, "OK");
            resp.SetBody(content, "text/html; charset=UTF-8");
            return resp.Serialize();
        }
    }

    // Serve static files
    std::string filePath = docRoot_ + path;
    if (std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath)) {
        std::string ext = std::filesystem::path(filePath).extension().string();
        std::string contentType = GetMimeType(ext);

        std::ifstream file(filePath, std::ios::binary);
        if (file) {
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> fileContent(fileSize);
            file.read(fileContent.data(), fileSize);

            HttpResponse resp;
            resp.SetStatus(200, "OK");
            resp.body = std::string(fileContent.data(), fileSize);
            resp.AddHeader("Content-Type", contentType);
            resp.AddHeader("Content-Length", std::to_string(fileSize));
            return resp.Serialize();
        } else {
            HttpResponse resp;
            resp.SetStatus(500, "Internal Server Error");
            resp.SetBody("<html><body><h1>500 Internal Server Error</h1></body></html>");
            return resp.Serialize();
        }
    }

    HttpResponse resp;
    resp.SetStatus(404, "Not Found");
    resp.SetBody("<html><body><h1>404 Not Found</h1><p>The requested URL was not found.</p></body></html>");
    return resp.Serialize();
}

} // namespace WebServer
