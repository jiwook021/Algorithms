/**
 * @file main.cpp
 * @brief Driver program for the UnsafeQueue demo.
 *
 * Demonstrates data-race behavior with multiple consumer threads.
 *
 * Build:  make && ./main
 */

#include "UnsafeQueue.hpp"
#include <iostream>
#include <thread>
#include <chrono>

/**
 * @brief Producer routine.
 * @param queue The shared queue.
 * @param numItems Items to produce.
 */
static void Producer(UnsafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.Produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/**
 * @brief Consumer routine.
 * @param queue The shared queue.
 * @param numItems Items to consume.
 */
static void Consumer(UnsafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        int out;
        queue.Consume(out);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    constexpr int NUM_ITEMS = 20;
    UnsafeQueue unsafeQueue;

    std::thread producerThread(Producer, std::ref(unsafeQueue), NUM_ITEMS);
    std::thread consumerThread(Consumer, std::ref(unsafeQueue), NUM_ITEMS);
    std::thread consumer2Thread(Consumer, std::ref(unsafeQueue), NUM_ITEMS);

    producerThread.join();
    consumerThread.join();
    consumer2Thread.join();
    return 0;
}
