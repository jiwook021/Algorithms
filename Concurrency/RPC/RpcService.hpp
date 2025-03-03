/**
 * @file RpcService.hpp
 * @brief Declarations for a simple TCP socket RPC server and client.
 *
 * @details
 * RpcServer listens on a TCP port, accepts a request containing a function
 * name and string arguments, executes a registered function, and sends the
 * result back.
 *
 * RpcClient connects to the server, sends one function call request, and
 * returns the response.
 */

#ifndef RPC_SERVICE_HPP
#define RPC_SERVICE_HPP

#include <functional>
#include <map>
#include <string>
#include <vector>

/**
 * @class RpcServer
 * @brief TCP-based RPC server that dispatches registered functions.
 */
class RpcServer {
public:
    /**
     * @brief Construct and bind the server to the given port.
     * @param port TCP port to listen on.
     */
    explicit RpcServer(int port);

    /** @brief Close the server socket. */
    ~RpcServer();

    /**
     * @brief Register a function for remote invocation.
     * @param name Function name used in the wire protocol.
     * @param func Callable that takes string args and returns a string result.
     */
    void RegisterFunction(
        const std::string& name,
        std::function<std::string(const std::vector<std::string>&)> func);

    /**
     * @brief Run the server event loop. Blocks forever.
     *
     * Accepts one connection at a time, parses the request, dispatches the
     * registered function, and sends the response.
     */
    void Run();

private:
    int port_;          ///< Listening port.
    int serverSocket_;  ///< Server socket file descriptor.

    /// Map of registered function names to implementations.
    std::map<
        std::string,
        std::function<std::string(const std::vector<std::string>&)>>
        functions_;
};

/**
 * @class RpcClient
 * @brief TCP-based RPC client that calls remote functions.
 */
class RpcClient {
public:
    /**
     * @brief Construct a client targeting the given server.
     * @param serverIp Server IP address.
     * @param serverPort Server port.
     */
    RpcClient(const std::string& serverIp, int serverPort);

    /**
     * @brief Call a remote function.
     * @param functionName Name of the registered function.
     * @param args String arguments.
     * @return Result string from the server, or an error string.
     */
    std::string Call(
        const std::string& functionName,
        const std::vector<std::string>& args);

private:
    std::string serverIp_;  ///< Server IP address.
    int serverPort_;        ///< Server port.
};

#endif // RPC_SERVICE_HPP
