/**
 * @file Sjf.cpp
 * @brief Implementation of the SJF scheduling algorithm.
 */

#include "Sjf.hpp"
#include <algorithm>
#include <functional>

namespace Sjf {

ScheduleResult ComputeSchedule(std::vector<Process> processes) {
    // Sort by arrival time.
    std::sort(processes.begin(), processes.end(),
              [](const Process& a, const Process& b) {
                  return a.arrivalTime < b.arrivalTime;
              });

    // Min-heap ordered by burst time.
    auto cmp = [](const Process& a, const Process& b) {
        return a.burstTime > b.burstTime;
    };
    std::priority_queue<Process, std::vector<Process>, decltype(cmp)> pq(cmp);

    ScheduleResult result;
    result.processResults.reserve(processes.size());

    int currentTime = 0;
    int idx = 0;
    int completed = 0;
    int n = static_cast<int>(processes.size());
    double totalWaiting = 0.0;
    double totalTurnaround = 0.0;

    while (completed < n) {
        // Enqueue all processes that have arrived by currentTime.
        while (idx < n && processes[idx].arrivalTime <= currentTime) {
            pq.push(processes[idx]);
            ++idx;
        }

        if (pq.empty()) {
            // CPU idle; advance to next arrival.
            currentTime = processes[idx].arrivalTime;
            continue;
        }

        auto proc = pq.top();
        pq.pop();

        int waitingTime = currentTime - proc.arrivalTime;
        currentTime += proc.burstTime;
        int turnaroundTime = currentTime - proc.arrivalTime;

        totalWaiting += waitingTime;
        totalTurnaround += turnaroundTime;

        result.processResults.push_back({proc.pid, waitingTime, turnaroundTime});
        ++completed;
    }

    auto dn = static_cast<double>(n);
    result.avgWaitingTime = (dn > 0) ? totalWaiting / dn : 0.0;
    result.avgTurnaroundTime = (dn > 0) ? totalTurnaround / dn : 0.0;

    return result;
}

} // namespace Sjf
