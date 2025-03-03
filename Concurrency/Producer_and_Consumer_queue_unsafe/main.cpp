#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <semaphore>
#include <chrono>
#include <optional>

// Producer-Consumer Queue without synchronization (Data Race Example)
class UnsafeQueue {
public:
    explicit UnsafeQueue()
:   m_availableItems(0) {}
    void produce(int item) {
        
        queue_.push(item);  // No synchronization, unsafe access
        std::cout << "Produced: " << item << "\n";
        m_availableItems.release();
    }

    void consume() {
        m_availableItems.acquire();
        if (!queue_.empty()) {  // No synchronization, unsafe check
            int item = queue_.front();  
            queue_.pop();
            std::cout << "Consumed: " << item << "\n";
        }
        std::cout<<"Entered Consume"<<std::endl;
    }

private:
    std::counting_semaphore<> m_availableItems;
    std::queue<int> queue_;
};

void producer(UnsafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void consumer(UnsafeQueue& queue, int numItems) {
    int consumed = 0;
    while (consumed < numItems) {
        queue.consume();
        ++consumed;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    constexpr int numItems = 20;
    UnsafeQueue unsafeQueue;

    std::thread producerThread(producer, std::ref(unsafeQueue), numItems);
    std::thread consumerThread(consumer, std::ref(unsafeQueue), numItems);
    std::thread consumer2Thread(consumer, std::ref(unsafeQueue), numItems);

    producerThread.join();
    consumerThread.join();
    return 0;
}
