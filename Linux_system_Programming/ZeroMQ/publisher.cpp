#include <Zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    Zmq::context_t Context{1};
    Zmq::socket_t Publisher{Context, Zmq::socket_type::pub};

    Publisher.bind("tcp://*:5555");  // Bind to port 5555 on localhost

    int count = 0;

    while (true) {
        std::string Message = "Hello ZeroMQ " + std::to_string(count++);

        Zmq::message_t ZmqMessage(Message.begin(), Message.end());
        Publisher.send(ZmqMessage, Zmq::send_flags::none);

        std::cout << "[Publisher] Sent: " << Message << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
