#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

// Subscriber base class (interface)
class Subscriber {
public:
    virtual ~Subscriber() {} // Virtual destructor for proper cleanup
    virtual void update(const std::string& message) = 0; // Pure virtual method
};

// Publisher class
class Publisher {
private:
    std::vector<Subscriber*> subscribers; // List of subscribers

public:
    // Add a subscriber to the list
    void subscribe(Subscriber* subscriber) {
        subscribers.push_back(subscriber);
    }

    // Remove a subscriber from the list
    void unsubscribe(Subscriber* subscriber) {
        auto it = std::find(subscribers.begin(), subscribers.end(), subscriber);
        if (it != subscribers.end()) {
            subscribers.erase(it);
        }
    }

    // Notify all subscribers with a message
    void notify(const std::string& message) {
        for (auto subscriber : subscribers) {
            subscriber->update(message);
        }
    }
};

// Concrete subscriber: ConsoleSubscriber
class ConsoleSubscriber : public Subscriber {
public:
    void update(const std::string& message) override {
        std::cout << "Console: " << message << std::endl;
    }
};

// Concrete subscriber: FileSubscriber
class FileSubscriber : public Subscriber {
public:
    void update(const std::string& message) override {
        // Simulate file logging by printing to console
        // In a real application, this would write to a file
        std::cout << "File: " << message << std::endl;
    }
};

// Main function to demonstrate the pattern
int main() {
    // Create publisher and subscribers
    Publisher publisher;
    ConsoleSubscriber consoleSub;
    FileSubscriber fileSub;

    // Subscribe both subscribers to the publisher
    publisher.subscribe(&consoleSub);
    publisher.subscribe(&fileSub);

    // Notify all subscribers of an event
    std::cout << "First notification:\n";
    publisher.notify("Event occurred!");

    // Unsubscribe the file subscriber
    publisher.unsubscribe(&fileSub);

    // Notify again, only console subscriber remains
    std::cout << "\nSecond notification (after unsubscribing FileSubscriber):\n";
    publisher.notify("Another event occurred!");

    return 0;
}