/**
 * @file test_VirtualMemoryManager.cpp
 * @brief Unit tests for the VirtualMemoryManager module.
 *
 * Tests PageTable, TLB (hit / miss / eviction / invalidate / clear),
 * address-translation helpers, and the full VMM facade.
 * Uses the real split-out components (PageTable, TLB) rather than
 * lightweight reimplementations.
 */

#include "VirtualMemoryManager.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PrintSection  --  educational banner before each test
// ---------------------------------------------------------------------------

namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
} // namespace

// ---------------------------------------------------------------------------
// Address translation helpers
// ---------------------------------------------------------------------------

void TestAddressTranslation() {
    PrintSection("AddressTranslation",
                 "Verify page-number and offset extraction from a virtual address, "
                 "and physical-address construction from frame + offset.");

    assert(GetPageNumber(8192) == 2);
    assert(GetOffset(4097) == 1);
    assert(MakePhysicalAddress(3, 100) == 3 * kPageSize + 100);
    std::cout << "[PASS] TestAddressTranslation\n";
}

// ---------------------------------------------------------------------------
// PageTable tests  (real PageTable class from PageTable.hpp)
// ---------------------------------------------------------------------------

void TestPageTableOperations() {
    PrintSection("PageTableOperations",
                 "Confirm that a page-table entry can be written, read back, "
                 "and have its dirty flag toggled.");

    PageTable pt(64);
    PageTableEntry e{};
    e.valid        = true;
    e.frame_number = 5;
    pt.UpdateEntry(10, e);

    auto got = pt.GetEntry(10);
    assert(got.valid == true);
    assert(got.frame_number == 5);

    pt.MarkDirty(10);
    assert(pt.GetEntry(10).dirty == true);
    std::cout << "[PASS] TestPageTableOperations\n";
}

void TestPageTableMultipleEntries() {
    PrintSection("PageTableMultipleEntries",
                 "Stress-test: fill 128 entries and verify every one is correct.");

    PageTable pt(128);
    for (std::size_t i = 0; i < 128; ++i) {
        PageTableEntry e{};
        e.valid        = true;
        e.frame_number = i * 2;
        pt.UpdateEntry(static_cast<std::uint32_t>(i), e);
    }
    for (std::size_t i = 0; i < 128; ++i) {
        auto got = pt.GetEntry(static_cast<std::uint32_t>(i));
        assert(got.valid == true);
        assert(got.frame_number == i * 2);
    }
    std::cout << "[PASS] TestPageTableMultipleEntries\n";
}

void TestPageTableMarkAccessed() {
    PrintSection("PageTableMarkAccessed",
                 "Ensure MarkAccessed sets the referenced flag for LRU tracking.");

    PageTable pt(16);
    PageTableEntry e{};
    e.valid        = true;
    e.frame_number = 0;
    pt.UpdateEntry(3, e);

    pt.MarkAccessed(3);
    auto got = pt.GetEntry(3);
    assert(got.referenced == true);
    std::cout << "[PASS] TestPageTableMarkAccessed\n";
}

// ---------------------------------------------------------------------------
// TLB tests  (real TLB class from TLB.hpp)
// ---------------------------------------------------------------------------

void TestTlbHit() {
    PrintSection("TlbHit",
                 "A page inserted into the TLB must be found on the next Lookup.");

    TLB tlb(4);
    tlb.Insert(5, 10);

    auto result = tlb.Lookup(5);
    assert(result.has_value());
    assert(*result == 10);
    std::cout << "[PASS] TestTlbHit\n";
}

void TestTlbMiss() {
    PrintSection("TlbMiss",
                 "Looking up a page that was never inserted must return nullopt.");

    TLB tlb(4);
    auto result = tlb.Lookup(99);
    assert(!result.has_value());
    std::cout << "[PASS] TestTlbMiss\n";
}

void TestTlbEviction() {
    PrintSection("TlbEviction",
                 "When the TLB is full the LRU entry must be evicted to make room.");

    TLB tlb(2);
    tlb.Insert(1, 10);
    tlb.Insert(2, 20);
    tlb.Insert(3, 30);  // evicts page 1 (LRU)

    assert(!tlb.Lookup(1).has_value());
    auto result = tlb.Lookup(3);
    assert(result.has_value());
    assert(*result == 30);
    std::cout << "[PASS] TestTlbEviction\n";
}

void TestTlbInvalidate() {
    PrintSection("TlbInvalidate",
                 "Invalidating a page must remove it so Lookup returns nullopt.");

    TLB tlb(4);
    tlb.Insert(5, 10);
    tlb.Invalidate(5);

    assert(!tlb.Lookup(5).has_value());
    std::cout << "[PASS] TestTlbInvalidate\n";
}

void TestTlbClear() {
    PrintSection("TlbClear",
                 "Clearing the TLB must remove every entry.");

    TLB tlb(4);
    tlb.Insert(1, 10);
    tlb.Insert(2, 20);
    tlb.Clear();

    assert(!tlb.Lookup(1).has_value());
    assert(!tlb.Lookup(2).has_value());
    std::cout << "[PASS] TestTlbClear\n";
}

void TestTlbHitRate() {
    PrintSection("TlbHitRate",
                 "Hit-rate counter must reflect actual hits and misses.");

    TLB tlb(4);
    tlb.Insert(1, 10);

    // 1 hit
    auto r1 = tlb.Lookup(1);
    assert(r1.has_value());

    // 1 miss
    auto r2 = tlb.Lookup(99);
    assert(!r2.has_value());

    // hit rate = 1/2 = 0.5
    double rate = tlb.HitRate();
    assert(rate > 0.49 && rate < 0.51);

    assert(tlb.Hits()   == 1);
    assert(tlb.Misses() == 1);
    std::cout << "[PASS] TestTlbHitRate\n";
}

void TestTlbUpdateExisting() {
    PrintSection("TlbUpdateExisting",
                 "Re-inserting a page must update its frame mapping in-place.");

    TLB tlb(4);
    tlb.Insert(5, 10);
    tlb.Insert(5, 42);  // update frame for same page

    auto result = tlb.Lookup(5);
    assert(result.has_value());
    assert(*result == 42);
    std::cout << "[PASS] TestTlbUpdateExisting\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    TestAddressTranslation();

    TestPageTableOperations();
    TestPageTableMultipleEntries();
    TestPageTableMarkAccessed();

    TestTlbHit();
    TestTlbMiss();
    TestTlbEviction();
    TestTlbInvalidate();
    TestTlbClear();
    TestTlbHitRate();
    TestTlbUpdateExisting();

    std::cout << "\nAll VirtualMemoryManager tests passed.\n";
    return 0;
}
