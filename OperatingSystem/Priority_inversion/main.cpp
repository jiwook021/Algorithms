/**
 * @file main.cpp
 * @brief Driver for the PriorityInversion module.
 */

#include "PriorityInversion.hpp"

// Defined in PriorityInversion.cpp
namespace PriorityInversion {
    void InitPiMutex();
    void DestroyPiMutex();
}

int main() {
    using namespace PriorityInversion;

    InitPiMutex();

    pthread_t thrLow, thrMed, thrHigh;
    constexpr int SCHEDULER_POLICY = SCHED_FIFO;

    CheckPthread(pthread_create(&thrLow, nullptr, LowPriorityTask, nullptr),
                 "pthread_create low");
    CheckPthread(pthread_create(&thrMed, nullptr, MediumPriorityTask, nullptr),
                 "pthread_create med");
    CheckPthread(pthread_create(&thrHigh, nullptr, HighPriorityTask, nullptr),
                 "pthread_create high");

    SetThreadPriority(thrLow,  SCHEDULER_POLICY, 10);
    SetThreadPriority(thrMed,  SCHEDULER_POLICY, 50);
    SetThreadPriority(thrHigh, SCHEDULER_POLICY, 80);

    pthread_join(thrLow,  nullptr);
    pthread_join(thrMed,  nullptr);
    pthread_join(thrHigh, nullptr);

    DestroyPiMutex();
    return 0;
}
