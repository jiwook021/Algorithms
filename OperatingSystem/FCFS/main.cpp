/**
 * @file main.cpp
 * @brief Driver for the FCFS scheduling module.
 */

#include "Fcfs.hpp"
#include <iostream>
#include <cstdio>

int main() {
    int numProcesses = 0;
    std::cout << "Enter number of processes: ";
    if (!(std::cin >> numProcesses) || numProcesses <= 0) {
        std::cerr << "Invalid number of processes\n";
        return 1;
    }

    std::vector<Fcfs::Process> processes;
    processes.reserve(numProcesses);

    std::cout << "Enter burst times:\n";
    for (int i = 0; i < numProcesses; ++i) {
        int burst = 0;
        if (!(std::cin >> burst)) {
            std::cerr << "Invalid burst time input\n";
            return 1;
        }
        processes.push_back({i + 1, 0, burst});
    }

    auto result = Fcfs::ComputeSchedule(processes);

    std::printf("%-10s %-12s %-14s %-16s\n",
                "Process", "Duration", "Waiting Time", "Turnaround Time");

    for (size_t i = 0; i < result.processResults.size(); ++i) {
        const auto& r = result.processResults[i];
        std::printf("%-10d %-12d %-14d %-16d\n",
                    r.pid, processes[i].burstTime,
                    r.waitingTime, r.turnaroundTime);
    }

    std::printf("Average Waiting Time: %.2f\n", result.avgWaitingTime);
    std::printf("Average Turnaround Time: %.2f\n", result.avgTurnaroundTime);

    return 0;
}
