#include <Zmq.hpp>
#include <string>
#include <iostream>

int main() {
    Zmq::context_t Context{1};
    Zmq::socket_t Subscriber{Context, Zmq::socket_type::sub};

    Subscriber.connect("tcp://localhost:5555");  // Connect to the publisher

    Subscriber.set(Zmq::sockopt::subscribe, "");  // Subscribe to all topics

    while (true) {
        Zmq::message_t ZmqMessage;
        Subscriber.recv(ZmqMessage, Zmq::recv_flags::none);

        std::string Message(static_cast<char*>(ZmqMessage.data()), ZmqMessage.size());
        std::cout << "[Subscriber] Received: " << Message << std::endl;
    }

    return 0;
}
