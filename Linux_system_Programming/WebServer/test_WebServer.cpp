#include "WebServer.hpp"
#include <cassert>
#include <iostream>

using namespace WebServer;

// ============================================================================
// HTTP Request Parsing Tests
// ============================================================================

void TestParseSimpleGet() {
    HttpRequest req;
    bool ok = ParseHttpRequest("GET /index.html HTTP/1.1\r\nHost: localhost\r\n\r\n", req);
    assert(ok);
    assert(req.method == "GET");
    assert(req.path == "/index.html");
    assert(req.httpVersion == "HTTP/1.1");
    assert(req.headers["Host"] == "localhost");
    std::cout << "[PASS] TestParseSimpleGet" << std::endl;
}

void TestParseGetWithQueryParams() {
    HttpRequest req;
    bool ok = ParseHttpRequest("GET /search?q=hello&lang=en HTTP/1.1\r\n\r\n", req);
    assert(ok);
    assert(req.path == "/search");
    assert(req.queryParams["q"] == "hello");
    assert(req.queryParams["lang"] == "en");
    std::cout << "[PASS] TestParseGetWithQueryParams" << std::endl;
}

void TestParsePost() {
    HttpRequest req;
    std::string raw = "POST /api/data HTTP/1.1\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: 13\r\n"
                      "\r\n"
                      "{\"key\":\"val\"}";
    bool ok = ParseHttpRequest(raw, req);
    assert(ok);
    assert(req.method == "POST");
    assert(req.path == "/api/data");
    assert(req.headers["Content-Type"] == "application/json");
    std::cout << "[PASS] TestParsePost" << std::endl;
}

void TestParseInvalid() {
    HttpRequest req;
    bool ok = ParseHttpRequest("", req);
    assert(!ok);
    std::cout << "[PASS] TestParseInvalid" << std::endl;
}

void TestParseMalformedRequestLine() {
    HttpRequest req;
    bool ok = ParseHttpRequest("BROKEN\r\n\r\n", req);
    assert(!ok);
    std::cout << "[PASS] TestParseMalformedRequestLine" << std::endl;
}

void TestParseMultipleHeaders() {
    HttpRequest req;
    std::string raw = "GET / HTTP/1.1\r\n"
                      "Host: example.com\r\n"
                      "Accept: text/html\r\n"
                      "Connection: keep-alive\r\n"
                      "\r\n";
    bool ok = ParseHttpRequest(raw, req);
    assert(ok);
    assert(req.headers.size() == 3);
    assert(req.headers["Accept"] == "text/html");
    assert(req.headers["Connection"] == "keep-alive");
    std::cout << "[PASS] TestParseMultipleHeaders" << std::endl;
}

void TestParseRootPath() {
    HttpRequest req;
    bool ok = ParseHttpRequest("GET / HTTP/1.1\r\n\r\n", req);
    assert(ok);
    assert(req.path == "/");
    std::cout << "[PASS] TestParseRootPath" << std::endl;
}

// ============================================================================
// HTTP Response Tests
// ============================================================================

void TestResponseSerialize200() {
    HttpResponse resp;
    resp.SetStatus(200, "OK");
    resp.SetBody("Hello", "text/plain");
    std::string s = resp.Serialize();
    assert(s.find("HTTP/1.1 200 OK") != std::string::npos);
    assert(s.find("Content-Type: text/plain") != std::string::npos);
    assert(s.find("Content-Length: 5") != std::string::npos);
    assert(s.find("Hello") != std::string::npos);
    std::cout << "[PASS] TestResponseSerialize200" << std::endl;
}

void TestResponseSerialize404() {
    HttpResponse resp;
    resp.SetStatus(404, "Not Found");
    resp.SetBody("Not found");
    std::string s = resp.Serialize();
    assert(s.find("HTTP/1.1 404 Not Found") != std::string::npos);
    std::cout << "[PASS] TestResponseSerialize404" << std::endl;
}

void TestResponseAddHeader() {
    HttpResponse resp;
    resp.SetStatus(200, "OK");
    resp.AddHeader("X-Custom", "value123");
    resp.SetBody("ok");
    std::string s = resp.Serialize();
    assert(s.find("X-Custom: value123") != std::string::npos);
    std::cout << "[PASS] TestResponseAddHeader" << std::endl;
}

void TestResponseConnectionClose() {
    HttpResponse resp;
    resp.SetStatus(200, "OK");
    resp.SetBody("test");
    std::string s = resp.Serialize();
    assert(s.find("Connection: close") != std::string::npos);
    std::cout << "[PASS] TestResponseConnectionClose" << std::endl;
}

void TestResponseEmptyBody() {
    HttpResponse resp;
    resp.SetStatus(204, "No Content");
    std::string s = resp.Serialize();
    assert(s.find("204 No Content") != std::string::npos);
    std::cout << "[PASS] TestResponseEmptyBody" << std::endl;
}

// ============================================================================
// MIME Type Tests
// ============================================================================

void TestMimeTypeHtml() {
    assert(GetMimeType(".html") == "text/html");
    assert(GetMimeType(".htm") == "text/html");
    std::cout << "[PASS] TestMimeTypeHtml" << std::endl;
}

void TestMimeTypeCss() {
    assert(GetMimeType(".css") == "text/css");
    std::cout << "[PASS] TestMimeTypeCss" << std::endl;
}

void TestMimeTypeJs() {
    assert(GetMimeType(".js") == "application/javascript");
    std::cout << "[PASS] TestMimeTypeJs" << std::endl;
}

void TestMimeTypeJson() {
    assert(GetMimeType(".json") == "application/json");
    std::cout << "[PASS] TestMimeTypeJson" << std::endl;
}

void TestMimeTypeImages() {
    assert(GetMimeType(".jpg") == "image/jpeg");
    assert(GetMimeType(".jpeg") == "image/jpeg");
    assert(GetMimeType(".png") == "image/png");
    assert(GetMimeType(".gif") == "image/gif");
    assert(GetMimeType(".svg") == "image/svg+xml");
    std::cout << "[PASS] TestMimeTypeImages" << std::endl;
}

void TestMimeTypeUnknown() {
    assert(GetMimeType(".xyz") == "application/octet-stream");
    assert(GetMimeType("") == "application/octet-stream");
    std::cout << "[PASS] TestMimeTypeUnknown" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== WebServer Unit Tests ===" << std::endl;

    // Request parsing
    TestParseSimpleGet();
    TestParseGetWithQueryParams();
    TestParsePost();
    TestParseInvalid();
    TestParseMalformedRequestLine();
    TestParseMultipleHeaders();
    TestParseRootPath();

    // Response
    TestResponseSerialize200();
    TestResponseSerialize404();
    TestResponseAddHeader();
    TestResponseConnectionClose();
    TestResponseEmptyBody();

    // MIME types
    TestMimeTypeHtml();
    TestMimeTypeCss();
    TestMimeTypeJs();
    TestMimeTypeJson();
    TestMimeTypeImages();
    TestMimeTypeUnknown();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
