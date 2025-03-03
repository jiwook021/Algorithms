#include "RestApi.hpp"
#include <cassert>
#include <iostream>

using namespace RestApi;

// ============================================================================
// Json Tests
// ============================================================================

void TestJsonParseSimpleObject() {
    auto obj = Json::Parse("{\"name\":\"Alice\",\"age\":30}");
    assert(std::get<std::string>(obj["name"]) == "Alice");
    assert(std::get<int>(obj["age"]) == 30);
    std::cout << "[PASS] TestJsonParseSimpleObject" << std::endl;
}

void TestJsonParseBooleanAndNull() {
    auto obj = Json::Parse("{\"active\":true,\"deleted\":false,\"data\":null}");
    assert(std::get<bool>(obj["active"]) == true);
    assert(std::get<bool>(obj["deleted"]) == false);
    assert(std::holds_alternative<std::nullptr_t>(obj["data"]));
    std::cout << "[PASS] TestJsonParseBooleanAndNull" << std::endl;
}

void TestJsonParseDouble() {
    auto obj = Json::Parse("{\"pi\":3.14}");
    double val = std::get<double>(obj["pi"]);
    assert(val > 3.13 && val < 3.15);
    std::cout << "[PASS] TestJsonParseDouble" << std::endl;
}

void TestJsonStringify() {
    Json::JsonObject obj;
    obj["key"] = std::string("value");
    obj["num"] = 42;
    std::string result = Json::Stringify(obj);
    assert(result.find("\"key\":\"value\"") != std::string::npos);
    assert(result.find("\"num\":42") != std::string::npos);
    std::cout << "[PASS] TestJsonStringify" << std::endl;
}

void TestJsonParseEmptyBraces() {
    auto obj = Json::Parse("{}");
    assert(obj.empty());
    std::cout << "[PASS] TestJsonParseEmptyBraces" << std::endl;
}

void TestJsonParseInvalidReturnsEmpty() {
    auto obj = Json::Parse("not json");
    assert(obj.empty());
    std::cout << "[PASS] TestJsonParseInvalidReturnsEmpty" << std::endl;
}

void TestJsonRoundTrip() {
    Json::JsonObject original;
    original["msg"] = std::string("hello");
    original["count"] = 7;
    original["flag"] = true;
    std::string serialized = Json::Stringify(original);
    auto parsed = Json::Parse(serialized);
    assert(std::get<std::string>(parsed["msg"]) == "hello");
    assert(std::get<int>(parsed["count"]) == 7);
    assert(std::get<bool>(parsed["flag"]) == true);
    std::cout << "[PASS] TestJsonRoundTrip" << std::endl;
}

// ============================================================================
// HttpParser Tests
// ============================================================================

void TestParseGetRequest() {
    std::string raw = "GET /api/hello HTTP/1.1\r\nHost: localhost\r\n\r\n";
    auto req = HttpParser::Parse(raw);
    assert(req.has_value());
    assert(req->method == HttpMethod::GET);
    assert(req->path == "/api/hello");
    assert(req->headers.count("Host") > 0);
    std::cout << "[PASS] TestParseGetRequest" << std::endl;
}

void TestParsePostRequestWithBody() {
    std::string raw = "POST /api/data HTTP/1.1\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: 16\r\n"
                      "\r\n"
                      "{\"key\":\"value\"}";
    auto req = HttpParser::Parse(raw);
    assert(req.has_value());
    assert(req->method == HttpMethod::POST);
    assert(req->path == "/api/data");
    assert(req->body == "{\"key\":\"value\"}");
    std::cout << "[PASS] TestParsePostRequestWithBody" << std::endl;
}

void TestParseQueryParameters() {
    std::string raw = "GET /api/search?q=test&page=2 HTTP/1.1\r\n\r\n";
    auto req = HttpParser::Parse(raw);
    assert(req.has_value());
    assert(req->path == "/api/search");
    assert(req->queryParams["q"] == "test");
    assert(req->queryParams["page"] == "2");
    std::cout << "[PASS] TestParseQueryParameters" << std::endl;
}

void TestParseInvalidRequest() {
    std::string raw = "INVALID DATA";
    auto req = HttpParser::Parse(raw);
    assert(!req.has_value());
    std::cout << "[PASS] TestParseInvalidRequest" << std::endl;
}

void TestParseDeleteRequest() {
    std::string raw = "DELETE /api/item HTTP/1.1\r\n\r\n";
    auto req = HttpParser::Parse(raw);
    assert(req.has_value());
    assert(req->method == HttpMethod::DELETE);
    assert(req->path == "/api/item");
    std::cout << "[PASS] TestParseDeleteRequest" << std::endl;
}

void TestParsePutRequest() {
    std::string raw = "PUT /api/item HTTP/1.1\r\n"
                      "Content-Length: 5\r\n"
                      "\r\n"
                      "hello";
    auto req = HttpParser::Parse(raw);
    assert(req.has_value());
    assert(req->method == HttpMethod::PUT);
    assert(req->body == "hello");
    std::cout << "[PASS] TestParsePutRequest" << std::endl;
}

// ============================================================================
// HttpFormatter Tests
// ============================================================================

void TestFormatOkResponse() {
    HttpResponse resp = {HttpStatus::OK,
                          {{"Content-Type", "text/plain"}},
                          "Hello"};
    std::string formatted = HttpFormatter::Format(resp);
    assert(formatted.find("HTTP/1.1 200 OK") != std::string::npos);
    assert(formatted.find("Hello") != std::string::npos);
    std::cout << "[PASS] TestFormatOkResponse" << std::endl;
}

void TestFormat404Response() {
    HttpResponse resp = {HttpStatus::NOT_FOUND,
                          {{"Content-Type", "application/json"}},
                          "{\"error\":\"not found\"}"};
    std::string formatted = HttpFormatter::Format(resp);
    assert(formatted.find("404 Not Found") != std::string::npos);
    std::cout << "[PASS] TestFormat404Response" << std::endl;
}

void TestFormatAutoContentLength() {
    HttpResponse resp = {HttpStatus::OK, {}, "body"};
    std::string formatted = HttpFormatter::Format(resp);
    assert(formatted.find("Content-Length: 4") != std::string::npos);
    std::cout << "[PASS] TestFormatAutoContentLength" << std::endl;
}

// ============================================================================
// Router Tests
// ============================================================================

void TestRouterFindsHandler() {
    Router router;
    router.RegisterHandler(HttpMethod::GET, "/test",
        [](const HttpRequest&) -> HttpResponse {
            return {HttpStatus::OK, {}, "found"};
        });
    HttpRequest req;
    req.method = HttpMethod::GET;
    req.path = "/test";
    auto resp = router.Route(req);
    assert(resp.status == HttpStatus::OK);
    assert(resp.body == "found");
    std::cout << "[PASS] TestRouterFindsHandler" << std::endl;
}

void TestRouterReturns404ForUnknown() {
    Router router;
    HttpRequest req;
    req.method = HttpMethod::GET;
    req.path = "/nonexistent";
    auto resp = router.Route(req);
    assert(resp.status == HttpStatus::NOT_FOUND);
    std::cout << "[PASS] TestRouterReturns404ForUnknown" << std::endl;
}

void TestRouterMethodMismatch() {
    Router router;
    router.RegisterHandler(HttpMethod::GET, "/only-get",
        [](const HttpRequest&) -> HttpResponse {
            return {HttpStatus::OK, {}, "ok"};
        });
    HttpRequest req;
    req.method = HttpMethod::POST;
    req.path = "/only-get";
    auto resp = router.Route(req);
    assert(resp.status == HttpStatus::NOT_FOUND);
    std::cout << "[PASS] TestRouterMethodMismatch" << std::endl;
}

void TestRouterHandlerException() {
    Router router;
    router.RegisterHandler(HttpMethod::GET, "/crash",
        [](const HttpRequest&) -> HttpResponse {
            throw std::runtime_error("boom");
        });
    HttpRequest req;
    req.method = HttpMethod::GET;
    req.path = "/crash";
    auto resp = router.Route(req);
    assert(resp.status == HttpStatus::INTERNAL_SERVER_ERROR);
    std::cout << "[PASS] TestRouterHandlerException" << std::endl;
}

// ============================================================================
// HttpMethod / HttpStatus string conversion tests
// ============================================================================

void TestHttpMethodToString() {
    assert(HttpMethodToString(HttpMethod::GET) == "GET");
    assert(HttpMethodToString(HttpMethod::POST) == "POST");
    assert(HttpMethodToString(HttpMethod::DELETE) == "DELETE");
    assert(HttpMethodToString(HttpMethod::UNKNOWN) == "UNKNOWN");
    std::cout << "[PASS] TestHttpMethodToString" << std::endl;
}

void TestHttpStatusToString() {
    assert(HttpStatusToString(HttpStatus::OK) == "200 OK");
    assert(HttpStatusToString(HttpStatus::NOT_FOUND) == "404 Not Found");
    assert(HttpStatusToString(HttpStatus::INTERNAL_SERVER_ERROR) == "500 Internal Server Error");
    std::cout << "[PASS] TestHttpStatusToString" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== RestApi Unit Tests ===" << std::endl;

    // Json
    TestJsonParseSimpleObject();
    TestJsonParseBooleanAndNull();
    TestJsonParseDouble();
    TestJsonStringify();
    TestJsonParseEmptyBraces();
    TestJsonParseInvalidReturnsEmpty();
    TestJsonRoundTrip();

    // HttpParser
    TestParseGetRequest();
    TestParsePostRequestWithBody();
    TestParseQueryParameters();
    TestParseInvalidRequest();
    TestParseDeleteRequest();
    TestParsePutRequest();

    // HttpFormatter
    TestFormatOkResponse();
    TestFormat404Response();
    TestFormatAutoContentLength();

    // Router
    TestRouterFindsHandler();
    TestRouterReturns404ForUnknown();
    TestRouterMethodMismatch();
    TestRouterHandlerException();

    // Helpers
    TestHttpMethodToString();
    TestHttpStatusToString();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
