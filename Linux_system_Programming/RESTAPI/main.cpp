#include "RestApi.hpp"

using namespace RestApi;

int main() {
    Logger::GetInstance().SetLevel(Logger::Level::DEBUG);
    Logger::GetInstance().SetLogFile("server.log");
    LOG_INFO("============= RESTful Web Server Start =============");

    WebServer server(8080, 4);

    // GET /api/hello
    server.RegisterHandler(HttpMethod::GET, "/api/hello",
        [](const HttpRequest& request) -> HttpResponse {
            Json::JsonObject responseData;
            responseData["message"] = std::string("Hello, World!");
            return {HttpStatus::OK,
                    {{"Content-Type", "application/json"}},
                    Json::Stringify(responseData)};
        });

    // GET /api/echo
    server.RegisterHandler(HttpMethod::GET, "/api/echo",
        [](const HttpRequest& request) -> HttpResponse {
            Json::JsonObject responseData;
            for (const auto& [key, value] : request.queryParams)
                responseData[key] = value;
            return {HttpStatus::OK,
                    {{"Content-Type", "application/json"}},
                    Json::Stringify(responseData)};
        });

    // POST /api/data
    server.RegisterHandler(HttpMethod::POST, "/api/data",
        [](const HttpRequest& request) -> HttpResponse {
            auto it = request.headers.find("Content-Type");
            if (it == request.headers.end() || it->second != "application/json") {
                return {HttpStatus::BAD_REQUEST,
                        {{"Content-Type", "application/json"}},
                        "{\"error\": \"Expected application/json\"}"};
            }
            Json::JsonObject requestData;
            try {
                requestData = Json::Parse(request.body);
            } catch (...) {
                return {HttpStatus::BAD_REQUEST,
                        {{"Content-Type", "application/json"}},
                        "{\"error\": \"Invalid JSON data\"}"};
            }
            Json::JsonObject responseData;
            for (const auto& [key, value] : requestData)
                responseData[key] = value;
            responseData["timestamp"] = std::to_string(std::time(nullptr));
            return {HttpStatus::OK,
                    {{"Content-Type", "application/json"}},
                    Json::Stringify(responseData)};
        });

    if (!server.Start()) {
        LOG_FATAL("Server start failed");
        return 1;
    }

    LOG_INFO("Server is running. Press Enter to stop...");
    std::cin.get();

    server.Stop();
    LOG_INFO("============= RESTful Web Server End =============");
    return 0;
}
