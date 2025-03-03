#include <gtest/gtest.h>

#include "SelfOrganizingList.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

TEST(SelfOrganizingListTest, InsertAndSize) {
    SelfOrganizingList<int> list;
    EXPECT_TRUE(list.Empty());
    EXPECT_EQ(list.Size(), 0u);

    list.Insert(1, 10);
    list.Insert(2, 20);
    list.Insert(3, 30);

    EXPECT_EQ(list.Size(), 3u);
    EXPECT_FALSE(list.Empty());
}

TEST(SelfOrganizingListTest, InsertOrdersByHighestPriorityFirst) {
    SelfOrganizingList<std::string> list;

    list.Insert("low", 10);
    list.Insert("high", 50);
    list.Insert("middle", 25);

    const std::vector<std::string> expected_items = {"high", "middle", "low"};
    const std::vector<std::pair<std::string, int>> expected_priorities = {
        {"high", 50},
        {"middle", 25},
        {"low", 10},
    };

    EXPECT_EQ(list.ToVector(), expected_items);
    EXPECT_EQ(list.ToPriorityVector(), expected_priorities);
}

TEST(SelfOrganizingListTest, EqualPrioritiesKeepOlderItemsFirst) {
    SelfOrganizingList<std::string> list;

    list.Insert("A", 10);
    list.Insert("B", 10);
    list.Insert("C", 10);

    const std::vector<std::string> expected = {"A", "B", "C"};
    EXPECT_EQ(list.ToVector(), expected);
}

TEST(SelfOrganizingListTest, UpdatePriorityMovesItemsHigherAndLower) {
    SelfOrganizingList<std::string> list;

    list.Insert("A", 10);
    list.Insert("B", 20);
    list.Insert("C", 30);

    EXPECT_TRUE(list.UpdatePriority("A", 40));

    const std::vector<std::string> after_raise = {"A", "C", "B"};
    EXPECT_EQ(list.ToVector(), after_raise);

    EXPECT_TRUE(list.UpdatePriority("A", 5));

    const std::vector<std::string> after_lower = {"C", "B", "A"};
    EXPECT_EQ(list.ToVector(), after_lower);
}

TEST(SelfOrganizingListTest, UpdatePriorityReturnsFalseForMissingItem) {
    SelfOrganizingList<std::string> list;

    list.Insert("A", 10);
    list.Insert("B", 20);

    EXPECT_FALSE(list.UpdatePriority("Z", 99));

    const std::vector<std::string> expected = {"B", "A"};
    EXPECT_EQ(list.ToVector(), expected);
}

TEST(SelfOrganizingListTest, UpdatePriorityTargetsFirstMatchingDuplicate) {
    SelfOrganizingList<std::string> list;

    list.Insert("task", 10);
    list.Insert("task", 20);
    list.Insert("other", 15);

    EXPECT_TRUE(list.UpdatePriority("task", 5));

    const std::vector<std::pair<std::string, int>> expected = {
        {"other", 15},
        {"task", 10},
        {"task", 5},
    };

    EXPECT_EQ(list.ToPriorityVector(), expected);
}

TEST(SelfOrganizingListTest, FindExistingAndMissingWithoutReordering) {
    SelfOrganizingList<std::string> list;

    list.Insert("low", 10);
    list.Insert("high", 50);
    list.Insert("middle", 25);

    const std::vector<std::string> before = list.ToVector();
    const SelfOrganizingList<std::string>& const_list = list;

    EXPECT_TRUE(const_list.Find("low"));
    EXPECT_TRUE(const_list.Find("high"));
    EXPECT_FALSE(const_list.Find("missing"));
    EXPECT_EQ(list.ToVector(), before);
}

TEST(SelfOrganizingListTest, FindOnEmptyList) {
    const SelfOrganizingList<int> list;
    EXPECT_FALSE(list.Find(42));
}

TEST(SelfOrganizingListTest, PopRemovesHighestPriorityFrontItem) {
    SelfOrganizingList<std::string> list;

    list.Insert("telemetry", 10);
    list.Insert("watchdog", 80);
    list.Insert("interrupt", 100);

    EXPECT_EQ(list.Pop(), "interrupt");
    EXPECT_EQ(list.Size(), 2u);
    EXPECT_FALSE(list.Find("interrupt"));

    const std::vector<std::string> expected = {"watchdog", "telemetry"};
    EXPECT_EQ(list.ToVector(), expected);
}

TEST(SelfOrganizingListTest, PopPreservesStableOrderForEqualPriorities) {
    SelfOrganizingList<std::string> list;

    list.Insert("sensor-A", 50);
    list.Insert("sensor-B", 50);
    list.Insert("sensor-C", 50);

    EXPECT_EQ(list.Pop(), "sensor-A");
    EXPECT_EQ(list.Pop(), "sensor-B");
    EXPECT_EQ(list.Pop(), "sensor-C");
    EXPECT_TRUE(list.Empty());
}

TEST(SelfOrganizingListTest, PopThrowsOnEmptyList) {
    SelfOrganizingList<int> list;

    EXPECT_THROW(list.Pop(), std::out_of_range);
}

TEST(EmbeddedSystemPriorityCases, PopRunsCriticalWorkBeforeLowPriorityWork) {
    SelfOrganizingList<std::string> scheduler;

    scheduler.Insert("telemetry upload", 10);
    scheduler.Insert("sensor sampling", 40);
    scheduler.Insert("watchdog heartbeat", 80);
    scheduler.Insert("interrupt service routine", 90);
    scheduler.Insert("thermal shutdown", 100);

    std::vector<std::string> execution_order;
    while (!scheduler.Empty()) {
        execution_order.push_back(scheduler.Pop());
    }

    const std::vector<std::string> expected = {
        "thermal shutdown",
        "interrupt service routine",
        "watchdog heartbeat",
        "sensor sampling",
        "telemetry upload",
    };

    EXPECT_EQ(execution_order, expected);
}
