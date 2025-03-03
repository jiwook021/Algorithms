/**
 * @file quick_sort_test.cpp
 * @brief Unit tests for quick_sort (quick_sort.hpp).
 *
 * Compile:
 *   g++ -std=c++23 -Wall -Wextra quick_sort_test.cpp -lgtest -lgtest_main -pthread -o quick_sort_test
 */

#include "QuickSort.hpp"

#include <string>
#include <vector>
#include <gtest/gtest.h>

// =============================================================================
// Google Test
// =============================================================================

TEST(QuickSortTest, EmptyRange) {
    std::vector<int> v;
    QuickSort(v.begin(), v.end());
    EXPECT_TRUE(v.empty());
}

TEST(QuickSortTest, SingleElement) {
    std::vector<int> v{42};
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, std::vector<int>{42});
}

TEST(QuickSortTest, AlreadySorted) {
    std::vector<int> v{1, 2, 3, 4, 5};
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(QuickSortTest, ReverseSorted) {
    std::vector<int> v{5, 4, 3, 2, 1};
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(QuickSortTest, RandomIntegers) {
    std::vector<int> v{3, 2, 4, 1, 7, 6, 5};
    QuickSort(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(QuickSortTest, Duplicates) {
    std::vector<int> v{5, 3, 3, 1, 5, 2, 1};
    QuickSort(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
    EXPECT_EQ(v, (std::vector<int>{1, 1, 2, 3, 3, 5, 5}));
}

TEST(QuickSortTest, AllEqual) {
    std::vector<int> v(10, 7);
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, std::vector<int>(10, 7));
}

TEST(QuickSortTest, TwoElements) {
    std::vector<int> v{2, 1};
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2}));
}

TEST(QuickSortTest, DescendingComparator) {
    std::vector<int> v{3, 1, 4, 1, 5, 9};
    QuickSort(v.begin(), v.end(), std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end(), std::greater<int>{}));
}

TEST(QuickSortTest, Strings) {
    std::vector<std::string> v{"banana", "apple", "cherry", "date"};
    QuickSort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<std::string>{"apple", "banana", "cherry", "date"}));
}

TEST(QuickSortTest, LargeArray) {
    std::vector<int> v(10000);
    for (int i = 0; i < 10000; ++i) v[i] = 10000 - i;
    QuickSort(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(QuickSortTest, CustomComparator) {
    struct Person { std::string Name; int Age; };
    std::vector<Person> People{{"Bob", 30}, {"Alice", 25}, {"Charlie", 35}};
    QuickSort(People.begin(), People.end(),
               [](const Person& a, const Person& b) { return a.Age < b.Age; });
    EXPECT_EQ(People[0].Name, "Alice");
    EXPECT_EQ(People[1].Name, "Bob");
    EXPECT_EQ(People[2].Name, "Charlie");
}
