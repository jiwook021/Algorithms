// Include the DBus C++ library header for DBus functionality
#include <dbus-c++-1/dbus-c++/dbus.h>
// Include standard C++ headers for string handling and I/O
#include <string>
#include <iostream>

// Define the Calculator service class, inheriting from DBus::ObjectAdaptor to expose methods over DBus
class Calculator : public DBus::ObjectAdaptor {
public:
    // Constructor: Registers the object with the DBus connection at a specific object path
    Calculator(DBus::Connection &connection)
        : DBus::ObjectAdaptor(connection, "/com/example/Calculator") {}

    // Method to add two integers and return the result
    // Exposed over DBus as 'add'
    int32_t add(const int32_t &a, const int32_t &b) {
        return a + b;
    }

    // Method to subtract one integer from another and return the result
    // Exposed over DBus as 'subtract'
    int32_t subtract(const int32_t &a, const int32_t &b) {
        return a - b;
    }
};

// Main function to set up and run the DBus service
int main() {
    // Initialize the DBus dispatcher, which manages the event loop for handling DBus messages
    DBus::BusDispatcher dispatcher;
    // Set this dispatcher as the default for DBus operations
    DBus::default_dispatcher = &dispatcher;

    // Establish a connection to the session bus (the bus for user-specific applications)
    DBus::Connection conn = DBus::Connection::SessionBus();

    // Request a well-known name for the service on the bus
    // This name (com.example.Calculator) is how clients will identify and connect to the service
    try {
        conn.request_name("com.example.Calculator");
    } catch (DBus::Error &e) {
        // If requesting the name fails (e.g., another instance is running), print an error and exit
        std::cerr << "Failed to request bus name: " << e.what() << std::endl;
        return 1;
    }

    // Create an instance of the Calculator service, registering it with the connection
    // This makes the 'add' and 'subtract' methods available at /com/example/Calculator
    Calculator calc(conn);

    // Print a message to indicate the service is running
    std::cout << "Calculator DBus service is running. Use a DBus client to call 'add' or 'subtract' methods." << std::endl;

    // Enter the event loop to listen for and handle incoming DBus method calls
    dispatcher.enter();

    // The program will run indefinitely until interrupted (e.g., with Ctrl+C)
    return 0;
}