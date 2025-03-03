/**
 * @file main.cpp
 * @brief Educational driver for a priority producer-consumer queue.
 */

#include "PriorityProducerConsumerQueue.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static std::mutex g_print_mutex;

template <typename... Args>
static void Print(Args&&... args) {
    std::scoped_lock lock(g_print_mutex);
    (std::cout << ... << std::forward<Args>(args));
    std::cout << std::flush;
}

static const char* PriorityName(int priority) {
    switch (priority) {
        case 0: return "LOW";
        case 1: return "NORMAL";
        case 2: return "HIGH";
        case 3: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

int main() {
    std::cout
        << "=============================================\n"
        << "  Priority Producer-Consumer Queue\n"
        << "=============================================\n\n"
        << "PURPOSE\n"
        << "  Producers submit uniquely identified work items:\n"
        << "    { id, priority, data }\n\n"
        << "  Consumers always receive the highest-priority waiting item.\n"
        << "  Items with the same priority remain FIFO.\n"
        << "  A waiting item can also be reprioritized by id.\n\n"
        << "PRIORITY LEVELS\n"
        << "  0 = LOW, 1 = NORMAL, 2 = HIGH, 3 = CRITICAL\n"
        << "  Larger number means higher priority.\n\n"
        << "HOW IT STAYS FAST\n"
        << "  The queue stores one FIFO bucket per priority level.\n"
        << "  An id map points directly to each waiting item.\n"
        << "  Consume() pops from the highest non-empty bucket.\n\n"
        << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Demo 1: Priority order and update by id
    // -----------------------------------------------------------------

    std::cout
        << "DEMO 1: Priority order with update by id\n\n";

    PriorityProducerConsumerQueue<std::string> triage_queue(5, 4);

    triage_queue.Produce(101, 0, "routine report");
    triage_queue.Produce(102, 2, "payment retry");
    triage_queue.Produce(103, 1, "email digest");

    std::cout << "  Produced three jobs.\n"
              << "  Updating id=101 from LOW to CRITICAL.\n\n";

    triage_queue.UpdatePriority(101, 3);

    while (triage_queue.Size() > 0) {
        auto item = triage_queue.Consume();
        if (item) {
            std::cout << "  Consumed id=" << item->id
                      << " priority=" << PriorityName(item->priority)
                      << " data=\"" << item->data << "\"\n";
        }
    }

    // -----------------------------------------------------------------
    // Demo 2: Blocking in action
    // -----------------------------------------------------------------

    std::cout
        << "\n---------------------------------------------\n"
        << "DEMO 2: Bounded capacity still blocks producers\n"
        << "---------------------------------------------\n\n";

    PriorityProducerConsumerQueue<int> small_queue(3, 4);
    std::atomic<int> placed{0};

    std::jthread fast_producer([&](std::stop_token) {
        for (int i = 1; i <= 6; ++i) {
            Print("  Producer: placing id=", i,
                  " priority=", PriorityName(i % 4), "\n");
            small_queue.Produce(i, i % 4, i * 10);
            placed.store(i, std::memory_order_release);
            Print("  Producer: id=", i, " placed [queue: ",
                  small_queue.Size(), "/3]\n");
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Print("\n  Main: producer placed ", placed.load(),
          " items and is blocked because capacity is full.\n");
    Print("  Main: consuming items to free slots.\n\n");

    for (int i = 0; i < 6; ++i) {
        auto item = small_queue.Consume();
        if (item) {
            Print("  Consumer: id=", item->id,
                  " priority=", PriorityName(item->priority),
                  " data=", item->data, "\n");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fast_producer.join();

    // -----------------------------------------------------------------
    // Demo 3: Multiple producers and consumers
    // -----------------------------------------------------------------

    std::cout
        << "\n---------------------------------------------\n"
        << "DEMO 3: Multiple producers and consumers\n"
        << "---------------------------------------------\n\n";

    constexpr int kItemsPerProducer = 5;
    constexpr int kNumProducers = 2;
    constexpr int kTotalItems = kItemsPerProducer * kNumProducers;

    PriorityProducerConsumerQueue<int> work_queue(4, 4);
    std::atomic<int> consumed{0};

    auto producer = [&](std::stop_token, int producer_id) {
        for (int i = 0; i < kItemsPerProducer; ++i) {
            const int id = producer_id * 100 + i;
            const int priority = (producer_id + i) % 4;
            work_queue.Produce(id, priority, id);
            Print("  Producer ", producer_id, " queued id=", id,
                  " priority=", PriorityName(priority), "\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    };

    auto consumer = [&](std::stop_token, int consumer_id, int count) {
        for (int i = 0; i < count; ++i) {
            auto item = work_queue.Consume();
            if (item) {
                consumed.fetch_add(1, std::memory_order_relaxed);
                Print("  Consumer ", consumer_id, " handled id=", item->id,
                      " priority=", PriorityName(item->priority), "\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    };

    {
        std::vector<std::jthread> threads;
        for (int p = 0; p < kNumProducers; ++p) {
            threads.emplace_back(producer, p);
        }
        threads.emplace_back(consumer, 0, kTotalItems / 2);
        threads.emplace_back(consumer, 1, kTotalItems - kTotalItems / 2);
    }

    std::cout << "\n  Total consumed: " << consumed.load()
              << " / " << kTotalItems << "\n";

    return 0;
}
