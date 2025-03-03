/**
 * @file main.cpp
 * @brief Driver for the SJF scheduling module.
 */

#include "Sjf.hpp"
#include <iostream>
#include <cstdio>

int main() {
    int numProcesses = 0;
    std::cout << "Enter number of processes: ";
    if (!(std::cin >> numProcesses) || numProcesses <= 0) {
        std::cerr << "Invalid number of processes\n";
        return 1;
    }

    std::vector<Sjf::Process> processes;
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

    auto result = Sjf::ComputeSchedule(processes);

    std::printf("%-10s %-12s %-14s %-16s\n",
                "Process", "Duration", "Waiting Time", "Turnaround Time");

    for (const auto& r : result.processResults) {
        // Find original burst time
        int burst = 0;
        for (const auto& p : processes) {
            if (p.pid == r.pid) { burst = p.burstTime; break; }
        }
        std::printf("%-10d %-12d %-14d %-16d\n",
                    r.pid, burst, r.waitingTime, r.turnaroundTime);
    }

    std::printf("Average Waiting Time: %.2f\n", result.avgWaitingTime);
    std::printf("Average Turnaround Time: %.2f\n", result.avgTurnaroundTime);

    return 0;
}
