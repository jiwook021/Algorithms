#include <iostream>
#include <thread>
#include <semaphore>
#include <chrono>
#include <vector>
#include <mutex>
// Custom fixed-size circular queue implementation
class CircularQueue {
public:
    explicit CircularQueue(size_t size)
        : m_size(size), m_head(0), m_tail(0), m_count(0), m_data(size) {}

    bool enqueue(int item) {
        if (m_count == m_size) {
            return false; // Queue full
        }
        m_data[m_tail] = item;
        m_tail = (m_tail + 1) % m_size;
        ++m_count;
        return true;
    }

    int dequeue(int& item) {
        if (m_count == 0) {
            return -1; // Queue empty
        }
        item = m_data[m_head];
        m_head = (m_head + 1) % m_size;
        --m_count;
        return item;
    }

private:
    const size_t m_size;
    size_t m_head;
    size_t m_tail;
    size_t m_count;
    std::vector<int> m_data;
};

class SafeQueue {
public:
    explicit SafeQueue(size_t maxSize)
        : queue_(maxSize), m_availableItems(0), m_availableSlots(maxSize) {}

    void produce(int item) {
        m_availableSlots.acquire(); // wait if no slot available
        {
            queue_.enqueue(item);
            std::cout << "Produced: " << item << "\n";
        }
        m_availableItems.release(); // signal item available
    }

    void consume() {
        m_availableItems.acquire(); // wait if no item available
        int item;
        std::cout << "Consumed: " << queue_.dequeue(item) << "\n";
        m_availableSlots.release(); // signal slot available
    }

private:
    CircularQueue queue_;
    std::counting_semaphore<> m_availableItems;
    std::counting_semaphore<> m_availableSlots;
    std::mutex m_mutex;
};

void producer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    constexpr int queueSize = 5;
    constexpr int numItems = 20;

    SafeQueue queue(queueSize);
    std::thread producerThread(producer, std::ref(queue), numItems);
    std::thread consumerThread(consumer, std::ref(queue), numItems);

    producerThread.join();
    consumerThread.join();

    return 0;
}