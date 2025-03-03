/**
 * @file WebSocketClient.cpp
 * @brief Implementation of the WebSocket client.
 */

#include "WebSocketClient.hpp"
#include <sstream>

namespace WebSocket {

void Client::Connect(const std::string& host, const std::string& port) {
    tcp::resolver resolver(ioContext_);
    wsStream_ = std::make_unique<websocket::stream<tcp::socket>>(ioContext_);

    auto results = resolver.resolve(host, port);
    net::connect(wsStream_->next_layer(), results.begin(), results.end());

    std::string hostHeader = host + ":" + port;
    wsStream_->handshake(hostHeader, "/");
}

void Client::Send(const std::string& message) {
    wsStream_->write(net::buffer(message));
}

std::string Client::Receive() {
    beast::flat_buffer buffer;
    wsStream_->read(buffer);

    std::ostringstream oss;
    oss << beast::make_printable(buffer.data());
    return oss.str();
}

void Client::Close() {
    wsStream_->close(websocket::close_code::normal);
}

} // namespace WebSocket
