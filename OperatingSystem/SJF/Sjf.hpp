/**
 * @file Sjf.hpp
 * @brief Shortest Job First (non-preemptive) CPU scheduling algorithm
 * @details Selects the process with the shortest burst time among ready
 *          processes. Provably optimal for minimising average waiting time
 *          in the non-preemptive case.
 */

#pragma once

#include <vector>
#include <queue>
#include <algorithm>

namespace Sjf {

/// A process waiting to be scheduled.
struct Process {
    int pid;
    int arrivalTime;
    int burstTime;
};

/// Per-process scheduling result.
struct ProcessResult {
    int pid;
    int waitingTime;
    int turnaroundTime;
};

/// Aggregate scheduling result.
struct ScheduleResult {
    std::vector<ProcessResult> processResults;
    double avgWaitingTime;
    double avgTurnaroundTime;
};

/**
 * Compute the SJF (non-preemptive) schedule.
 *
 * @param processes  Input processes (copied).
 * @return ScheduleResult with per-process and average metrics.
 */
ScheduleResult ComputeSchedule(std::vector<Process> processes);

} // namespace Sjf
