/**
 * @file main.cpp
 * @brief Driver for SelfOrganizingList embedded-priority scheduling.
 */

#include "SelfOrganizingList.hpp"

#include <iostream>
#include <string>

template <typename T>
void Display(const SelfOrganizingList<T>& List) {
    for (const auto& Entry : List.ToPriorityVector()) {
        std::cout << Entry.first << "(priority " << Entry.second << ") -> ";
    }
    std::cout << "nullptr\n";
}

int main() {
    SelfOrganizingList<std::string> Scheduler;

    Scheduler.Insert("telemetry upload", 10);
    Scheduler.Insert("sensor sampling", 40);
    Scheduler.Insert("watchdog heartbeat", 80);
    Scheduler.Insert("interrupt service routine", 90);
    Scheduler.Insert("thermal shutdown", 100);

    std::cout << "=== Embedded Priority Work Scheduler ===\n";
    std::cout << "Queued work: ";
    Display(Scheduler);

    std::cout << std::boolalpha;
    std::cout << "Watchdog queued: "
              << Scheduler.Find("watchdog heartbeat") << '\n';

    Scheduler.UpdatePriority("sensor sampling", 95);
    std::cout << "After urgent sensor update: ";
    Display(Scheduler);

    std::cout << "Run order:\n";
    while (!Scheduler.Empty()) {
        std::cout << "  running " << Scheduler.Pop() << '\n';
    }

    return 0;
}
