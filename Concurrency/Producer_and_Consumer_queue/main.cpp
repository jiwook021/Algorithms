#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <semaphore>
#include <chrono>
#include <optional>

// Producer-Consumer Queue using C++20 Semaphores
class ProducerConsumerQueue {
public:
    explicit ProducerConsumerQueue(size_t maxSize)
        : m_maxSize(maxSize),
          m_availableSlots(maxSize),
          m_availableItems(0) {}
    
    // Thread-safe producer, write 
    void produce(int item) {
        m_availableSlots.acquire(); // Wait until there's space available

        std::scoped_lock lock(m_mutex);
        m_queue.push(item);
        std::cout << "Produced: " << item << "\n";

        m_availableItems.release(); // Signal an item is available
    }

    // Thread-safe consumer, read 
    std::optional<int> consume() {
        m_availableItems.acquire(); // Wait until there's an item to consume

        std::scoped_lock lock(m_mutex);
        if (m_queue.empty()) {
            return std::nullopt;
        }

        int item = m_queue.front();
        m_queue.pop();
        std::cout << "Consumed: " << item << "\n";

        m_availableSlots.release(); // Signal a slot is available
        return item;
    }

private:
    size_t m_maxSize;
    std::queue<int> m_queue;
    std::mutex m_mutex;
    std::counting_semaphore<> m_availableSlots;
    std::counting_semaphore<> m_availableItems;
};

void producer(ProducerConsumerQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void consumer(ProducerConsumerQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    constexpr int queueSize = 5;
    constexpr int numItems = 10;

    ProducerConsumerQueue pcQueue(queueSize);

    std::thread producerThread(producer, std::ref(pcQueue), numItems);
    std::thread consumerThread(consumer, std::ref(pcQueue), numItems);

    producerThread.join();
    consumerThread.join();

    return 0;
}
