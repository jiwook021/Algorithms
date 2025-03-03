/**
 * @file main.cpp
 * @brief Driver for VirtualMemoryManager.
 *
 * Exercises sequential, random, and locality-of-reference access patterns
 * with three page-replacement policies: FIFO, LRU, and Clock.
 */

#include "VirtualMemoryManager.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "=============================================\n"
                  << "  Virtual Memory Manager Simulation\n"
                  << "=============================================\n\n"

                  << "PURPOSE\n"
                  << "  This program simulates how an operating system manages\n"
                  << "  virtual memory.  Every process sees a large, contiguous\n"
                  << "  address space (4 GB), but physical RAM is much smaller\n"
                  << "  (16 MB).  The VMM translates virtual addresses to physical\n"
                  << "  ones, swapping pages to disk when RAM is full.\n\n"

                  << "HOW IT WORKS\n"
                  << "  1. A program reads/writes a virtual address.\n"
                  << "  2. The TLB (Translation Lookaside Buffer, " << kTlbSize << " entries)\n"
                  << "     is checked first for a cached page-to-frame mapping.\n"
                  << "  3. On TLB miss, the page table is consulted.\n"
                  << "  4. If the page is not in RAM (page fault), a free frame\n"
                  << "     is allocated — or a victim page is evicted to make room.\n"
                  << "  5. The page is loaded from swap space into\n"
                  << "     the frame, and the mapping is cached in the TLB.\n\n"

                  << "CONFIGURATION\n"
                  << "  Page size       : " << kPageSize / 1024 << " KB\n"
                  << "  Virtual memory  : " << (kVirtualMemorySize >> 30) << " GB  ("
                  << (kVirtualMemorySize / kPageSize) << " pages)\n"
                  << "  Physical memory : " << (kPhysicalMemorySize >> 20) << " MB  ("
                  << (kPhysicalMemorySize / kPageSize) << " frames)\n"
                  << "  Swap space      : " << (kSwapSize >> 20) << " MB\n"
                  << "  TLB entries     : " << kTlbSize << "\n\n"

                  << "PAGE REPLACEMENT POLICIES\n"
                  << "  When RAM is full, the VMM must pick a page to evict.\n"
                  << "  This program tests three classic algorithms:\n\n"

                  << "  FIFO  — Evict the page that was loaded first.\n"
                  << "          Simple but ignores access frequency.\n\n"
                  << "  LRU   — Evict the page that was accessed least recently.\n"
                  << "          Adapts to workload but costs more to track.\n\n"
                  << "  Clock — Approximates LRU with a circular buffer and\n"
                  << "          reference bits.  Cheaper than true LRU.\n\n"

                  << "ACCESS PATTERNS TESTED\n"
                  << "  Sequential — Addresses visited in order (stride-1).\n"
                  << "               Expect high TLB hit rate since consecutive\n"
                  << "               bytes share the same 4 KB page.\n\n"
                  << "  Random     — Addresses visited in shuffled order.\n"
                  << "               Stresses the TLB and page replacement;\n"
                  << "               expect more page faults.\n\n"
                  << "  Locality   — 90% of accesses target 5 'hot' pages.\n"
                  << "               Models real programs (loops, caches).\n"
                  << "               LRU/Clock should outperform FIFO here.\n"
                  << "---------------------------------------------\n\n";

        // ----- FIFO --------------------------------------------------------
        std::cout << "===== FIFO Page Replacement =====\n";
        {
            auto replacer = std::make_unique<FIFOReplacer>();
            VirtualMemoryManager vmm(
                kVirtualMemorySize / kPageSize,
                kPhysicalMemorySize,
                kSwapSize,
                kTlbSize,
                std::move(replacer));

            TestSequentialAccess(vmm);
            TestRandomAccess(vmm);
            TestLocalityOfReference(vmm);
        }

        // ----- LRU ---------------------------------------------------------
        std::cout << "\n===== LRU Page Replacement =====\n";
        {
            auto replacer = std::make_unique<LRUReplacer>();
            VirtualMemoryManager vmm(
                kVirtualMemorySize / kPageSize,
                kPhysicalMemorySize,
                kSwapSize,
                kTlbSize,
                std::move(replacer));

            TestSequentialAccess(vmm);
            TestRandomAccess(vmm);
            TestLocalityOfReference(vmm);
        }

        // ----- Clock -------------------------------------------------------
        std::cout << "\n===== Clock Page Replacement =====\n";
        {
            auto replacer = std::make_unique<ClockReplacer>(
                kPhysicalMemorySize / kPageSize);
            VirtualMemoryManager vmm(
                kVirtualMemorySize / kPageSize,
                kPhysicalMemorySize,
                kSwapSize,
                kTlbSize,
                std::move(replacer));

            TestSequentialAccess(vmm);
            TestRandomAccess(vmm);
            TestLocalityOfReference(vmm);
        }

        std::cout << "\nSimulation completed successfully.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
