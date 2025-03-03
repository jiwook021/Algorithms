/**
 * @file Fcfs.cpp
 * @brief Implementation of the FCFS scheduling algorithm.
 */

#include "Fcfs.hpp"
#include <algorithm>
#include <numeric>

namespace Fcfs {

ScheduleResult ComputeSchedule(std::vector<Process> processes) {
    // Sort by arrival time (stable to preserve insertion order on ties).
    std::sort(processes.begin(), processes.end(),
              [](const Process& a, const Process& b) {
                  return a.arrivalTime < b.arrivalTime;
              });

    ScheduleResult result;
    result.processResults.reserve(processes.size());

    int currentTime = 0;
    double totalWaiting = 0.0;
    double totalTurnaround = 0.0;

    for (const auto& proc : processes) {
        // If CPU is idle, advance time to the next arrival.
        if (currentTime < proc.arrivalTime) {
            currentTime = proc.arrivalTime;
        }

        int waitingTime = currentTime - proc.arrivalTime;
        currentTime += proc.burstTime;
        int turnaroundTime = currentTime - proc.arrivalTime;

        totalWaiting += waitingTime;
        totalTurnaround += turnaroundTime;

        result.processResults.push_back({proc.pid, waitingTime, turnaroundTime});
    }

    auto n = static_cast<double>(processes.size());
    result.avgWaitingTime = (n > 0) ? totalWaiting / n : 0.0;
    result.avgTurnaroundTime = (n > 0) ? totalTurnaround / n : 0.0;

    return result;
}

} // namespace Fcfs
