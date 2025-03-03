/**
 * @file WebSocketClient.hpp
 * @brief WebSocket client using Boost.Beast
 * @details Wraps Boost.Beast WebSocket functionality: connect, send, receive,
 *          and close a WebSocket connection to a remote server.
 */

#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/connect.hpp>

#include <string>

namespace WebSocket {

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
using tcp           = net::ip::tcp;

/// A synchronous WebSocket client.
class Client {
public:
    /// Connect to host:port, perform WebSocket handshake.
    void Connect(const std::string& host, const std::string& port);

    /// Send a text message.
    void Send(const std::string& message);

    /// Read a message from the server and return it.
    std::string Receive();

    /// Perform a clean WebSocket close.
    void Close();

private:
    net::io_context ioContext_;
    std::unique_ptr<websocket::stream<tcp::socket>> wsStream_;
};

} // namespace WebSocket
