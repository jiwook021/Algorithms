/**
 * @file main.cpp
 * @brief Driver program for the CircularQueue / SafeQueue demo.
 *
 * Demonstrates the semaphore-wrapped SafeQueue with a producer and consumer.
 *
 * Build:  make && ./main
 */

#include "CircularQueue.hpp"
#include <iostream>
#include <thread>
#include <chrono>

/**
 * @brief Producer routine.
 * @param queue The shared safe queue.
 * @param numItems Number of items to produce.
 */
static void Producer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.Produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/**
 * @brief Consumer routine.
 * @param queue The shared safe queue.
 * @param numItems Number of items to consume.
 */
static void Consumer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        int out;
        queue.Consume(out);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    constexpr int QUEUE_SIZE = 5;
    constexpr int NUM_ITEMS  = 20;

    SafeQueue queue(QUEUE_SIZE);
    std::thread p(Producer, std::ref(queue), NUM_ITEMS);
    std::thread c(Consumer, std::ref(queue), NUM_ITEMS);

    p.join();
    c.join();
    return 0;
}
