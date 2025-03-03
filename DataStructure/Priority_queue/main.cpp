/**
 * @file main.cpp
 * @brief Driver for PriorityQueue
 */

#include "PriorityQueue.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Priority Queue Demo ===\n\n";

    // --- Min-Heap: Earliest Deadline First Scheduling ---
    std::cout << "--- Min-Heap: RTOS Deadlines (lower tick = run sooner) ---\n";
    MinPriorityQueue<int> deadlineQueue;
    std::cout << "Control-loop jobs released with absolute deadlines:\n";
    for (int deadlineTick : {42, 16, 28, 10, 35}) {
        std::cout << "  Job queued       [deadline tick " << deadlineTick << "]\n";
        deadlineQueue.Push(deadlineTick);
    }
    std::cout << "Dispatch order:    ";
    while (!deadlineQueue.Empty()) {
        std::cout << deadlineQueue.Top() << " ";
        deadlineQueue.Pop();
    }
    std::cout << "\n\n";

    // --- Max-Heap: Interrupt Deferred Work ---
    std::cout << "--- Max-Heap: Deferred IRQ Work (higher priority = run first) ---\n";
    MaxPriorityQueue<int> irqWorkQueue;
    std::cout << "Deferred interrupt work queued by criticality:\n";
    for (int priority : {10, 40, 20, 50, 30}) {
        std::cout << "  Work item queued [priority " << priority << "]\n";
        irqWorkQueue.Push(priority);
    }
    std::cout << "Execution order:   ";
    while (!irqWorkQueue.Empty()) {
        std::cout << irqWorkQueue.Top() << " ";
        irqWorkQueue.Pop();
    }
    std::cout << "\n\n";

    // --- Min-Heap: Sensor Sampling Timer Queue ---
    std::cout << "--- Min-Heap: Sensor Timer Events (lower tick = fire first) ---\n";
    MinPriorityQueue<int> sensorTimerQueue;
    std::cout << "Sensor read events scheduled at RTOS ticks:\n";
    for (int sampleTick : {1000, 250, 500, 125, 750}) {
        std::cout << "  Timer armed      [tick " << sampleTick << "]\n";
        sensorTimerQueue.Push(sampleTick);
    }
    std::cout << "Fire order:        ";
    while (!sensorTimerQueue.Empty()) {
        std::cout << sensorTimerQueue.Top() << " ";
        sensorTimerQueue.Pop();
    }
    std::cout << "\n\n";

    // --- Construct from Range ---
    std::cout << "--- Construct Min-Heap from Boot-Time Alarm Deadlines ---\n";
    std::vector<int> alarmDeadlines = {88, 42, 95, 71, 63, 100, 55};
    std::cout << "Unordered alarms:  ";
    for (int deadline : alarmDeadlines) std::cout << deadline << " ";
    std::cout << "\n";

    MinPriorityQueue<int> alarmQueue(alarmDeadlines.begin(), alarmDeadlines.end());
    std::cout << "Alarm order:       ";
    while (!alarmQueue.Empty()) {
        std::cout << alarmQueue.Top() << " ";
        alarmQueue.Pop();
    }
    std::cout << "\n";

    return 0;
}
