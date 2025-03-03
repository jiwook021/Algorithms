/**
 * @file Fcfs.hpp
 * @brief First-Come-First-Served CPU scheduling algorithm
 * @details Simulates FCFS scheduling for processes with arrival and burst times.
 *          Computes per-process and average waiting/turnaround times.
 */

#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace Fcfs {

/// Represents a single process in the scheduling queue.
struct Process {
    int pid;
    int arrivalTime;
    int burstTime;
};

/// Holds the computed scheduling result for a single process.
struct ProcessResult {
    int pid;
    int waitingTime;
    int turnaroundTime;
};

/// Aggregate result returned by the scheduler.
struct ScheduleResult {
    std::vector<ProcessResult> processResults;
    double avgWaitingTime;
    double avgTurnaroundTime;
};

/**
 * Compute the FCFS schedule for the given processes.
 *
 * Processes are sorted by arrival time, then served in that order.
 * @param processes  Vector of Process structs (copied internally).
 * @return ScheduleResult with per-process and average metrics.
 */
ScheduleResult ComputeSchedule(std::vector<Process> processes);

} // namespace Fcfs
