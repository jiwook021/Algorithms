/**
 * @file test_Sjf.cpp
 * @brief Unit tests for the SJF scheduling module.
 */

#include "Sjf.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static constexpr double EPSILON = 0.01;

static bool Near(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}

/// SJF reorders by burst time: shortest first.
void TestBasicScheduling() {
    std::vector<Sjf::Process> procs = {
        {1, 0, 6},
        {2, 0, 2},
        {3, 0, 1}
    };
    auto result = Sjf::ComputeSchedule(procs);

    // Execution order: P3(1), P2(2), P1(6). Waits: 0, 1, 3.
    assert(Near(result.avgWaitingTime, (0.0 + 1.0 + 3.0) / 3.0));
    assert(result.processResults[0].pid == 3);
    assert(result.processResults[1].pid == 2);
    assert(result.processResults[2].pid == 1);
    std::cout << "[PASS] TestBasicScheduling\n";
}

/// Two processes: shortest runs first.
void TestShortestFirst() {
    std::vector<Sjf::Process> procs = {
        {1, 0, 10},
        {2, 0, 1}
    };
    auto result = Sjf::ComputeSchedule(procs);

    // P2 first (wait 0), P1 second (wait 1).
    assert(Near(result.avgWaitingTime, 0.5));
    assert(result.processResults[0].pid == 2);
    std::cout << "[PASS] TestShortestFirst\n";
}

/// Single process: trivial case.
void TestSingleProcess() {
    std::vector<Sjf::Process> procs = {{1, 0, 5}};
    auto result = Sjf::ComputeSchedule(procs);

    assert(result.avgWaitingTime == 0.0);
    assert(result.avgTurnaroundTime == 5.0);
    std::cout << "[PASS] TestSingleProcess\n";
}

/// Processes with staggered arrivals.
void TestStaggeredArrivals() {
    std::vector<Sjf::Process> procs = {
        {1, 0, 8},
        {2, 1, 1},
        {3, 2, 2}
    };
    auto result = Sjf::ComputeSchedule(procs);

    // At t=0 only P1 available -> runs P1 until t=8.
    // At t=8 P2 and P3 are ready. P2 (burst 1) runs first -> t=9.
    // P3 runs -> t=11.
    assert(result.processResults[0].pid == 1);
    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[1].pid == 2);
    assert(result.processResults[1].waitingTime == 7);
    assert(result.processResults[2].pid == 3);
    assert(result.processResults[2].waitingTime == 7);
    std::cout << "[PASS] TestStaggeredArrivals\n";
}

/// CPU idle gap between arrivals.
void TestIdleGap() {
    std::vector<Sjf::Process> procs = {
        {1, 0, 2},
        {2, 10, 3}
    };
    auto result = Sjf::ComputeSchedule(procs);

    assert(result.processResults[0].waitingTime == 0);
    assert(result.processResults[1].waitingTime == 0);
    assert(result.processResults[1].turnaroundTime == 3);
    std::cout << "[PASS] TestIdleGap\n";
}

int main() {
    TestBasicScheduling();
    TestShortestFirst();
    TestSingleProcess();
    TestStaggeredArrivals();
    TestIdleGap();
    std::cout << "\nAll SJF tests passed.\n";
    return 0;
}
