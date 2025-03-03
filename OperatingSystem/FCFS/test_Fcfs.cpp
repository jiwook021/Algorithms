/**
 * @file test_Fcfs.cpp
 * @brief Unit tests for the FCFS scheduling module.
 */

#include "Fcfs.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static constexpr double EPSILON = 0.01;

static bool Near(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}

/// Basic scheduling: three processes arriving at different times.
void TestBasicScheduling() {
    std::vector<Fcfs::Process> procs = {
        {1, 0, 5},
        {2, 1, 3},
        {3, 2, 1}
    };
    auto result = Fcfs::ComputeSchedule(procs);

    // P1 waits 0, P2 waits 4, P3 waits 6
    assert(Near(result.avgWaitingTime, (0.0 + 4.0 + 6.0) / 3.0));
    assert(result.processResults.size() == 3);
    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[1].waitingTime == 4);
    assert(result.processResults[2].waitingTime == 6);
    std::cout << "[PASS] TestBasicScheduling\n";
}

/// All processes arrive at time 0.
void TestAllArriveAtZero() {
    std::vector<Fcfs::Process> procs = {
        {1, 0, 3},
        {2, 0, 2},
        {3, 0, 1}
    };
    auto result = Fcfs::ComputeSchedule(procs);

    assert(Near(result.avgWaitingTime, (0.0 + 3.0 + 5.0) / 3.0));
    assert(result.processResults[0].turnaroundTime == 3);
    assert(result.processResults[1].turnaroundTime == 5);
    assert(result.processResults[2].turnaroundTime == 6);
    std::cout << "[PASS] TestAllArriveAtZero\n";
}

/// Single process - zero waiting, turnaround equals burst.
void TestSingleProcess() {
    std::vector<Fcfs::Process> procs = {{1, 0, 10}};
    auto result = Fcfs::ComputeSchedule(procs);

    assert(result.avgWaitingTime == 0.0);
    assert(result.avgTurnaroundTime == 10.0);
    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[0].turnaroundTime == 10);
    std::cout << "[PASS] TestSingleProcess\n";
}

/// Processes with gaps between arrivals (CPU goes idle).
void TestGapBetweenArrivals() {
    std::vector<Fcfs::Process> procs = {
        {1, 0, 2},
        {2, 10, 3}
    };
    auto result = Fcfs::ComputeSchedule(procs);

    // P1: wait=0, tat=2. P2: arrives at 10, CPU idle until 10, wait=0, tat=3
    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[1].waitingTime == 0);
    assert(result.processResults[1].turnaroundTime == 3);
    std::cout << "[PASS] TestGapBetweenArrivals\n";
}

/// Out-of-order arrival times are sorted correctly.
void TestOutOfOrderArrival() {
    std::vector<Fcfs::Process> procs = {
        {3, 5, 1},
        {1, 0, 4},
        {2, 2, 2}
    };
    auto result = Fcfs::ComputeSchedule(procs);

    // Sorted order: P1(0,4), P2(2,2), P3(5,1)
    // P1: wait=0, tat=4. P2: wait=2, tat=4. P3: wait=1, tat=2
    assert(result.processResults[0].pid == 1);
    assert(result.processResults[1].pid == 2);
    assert(result.processResults[2].pid == 3);
    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[1].waitingTime == 2);
    assert(result.processResults[2].waitingTime == 1);
    std::cout << "[PASS] TestOutOfOrderArrival\n";
}

int main() {
    TestBasicScheduling();
    TestAllArriveAtZero();
    TestSingleProcess();
    TestGapBetweenArrivals();
    TestOutOfOrderArrival();
    std::cout << "\nAll FCFS tests passed.\n";
    return 0;
}
