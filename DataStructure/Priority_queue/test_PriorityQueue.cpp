/**
 * @file test_PriorityQueue.cpp
 * @brief Unit tests for PriorityQueue (min-heap and max-heap)
 */

#include "PriorityQueue.hpp"
#include <gtest/gtest.h>
#include <vector>

// --- Min-Heap Tests ---

TEST(MinPriorityQueue, EmptyOnConstruction) {
    MinPriorityQueue<int> pq;
    EXPECT_TRUE(pq.Empty());
    EXPECT_EQ(pq.Size(), 0u);
}

TEST(MinPriorityQueue, PushAndTop) {
    MinPriorityQueue<int> pq;
    pq.Push(3);
    pq.Push(1);
    pq.Push(2);
    EXPECT_EQ(pq.Top(), 1);
    EXPECT_EQ(pq.Size(), 3u);
}

TEST(MinPriorityQueue, PopOrder) {
    MinPriorityQueue<int> pq;
    pq.Push(3);
    pq.Push(1);
    pq.Push(2);
    pq.Push(5);
    pq.Push(4);

    std::vector<int> result;
    while (!pq.Empty()) {
        result.push_back(pq.Top());
        pq.Pop();
    }
    EXPECT_EQ(result, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(MinPriorityQueue, ConstructFromRange) {
    std::vector<int> values = {5, 2, 8, 1, 9};
    MinPriorityQueue<int> pq(values.begin(), values.end());
    EXPECT_EQ(pq.Size(), 5u);
    EXPECT_EQ(pq.Top(), 1);
}

TEST(MinPriorityQueue, Clear) {
    MinPriorityQueue<int> pq;
    pq.Push(1);
    pq.Push(2);
    pq.Clear();
    EXPECT_TRUE(pq.Empty());
}

TEST(MinPriorityQueue, TopOnEmptyThrows) {
    MinPriorityQueue<int> pq;
    EXPECT_THROW(
        {
            const int& top = pq.Top();
            (void)top;
        },
        std::out_of_range);
}

TEST(MinPriorityQueue, PopOnEmptyThrows) {
    MinPriorityQueue<int> pq;
    EXPECT_THROW(pq.Pop(), std::out_of_range);
}

// --- Max-Heap Tests ---

TEST(MaxPriorityQueue, PopOrder) {
    MaxPriorityQueue<int> pq;
    pq.Push(3);
    pq.Push(1);
    pq.Push(5);
    pq.Push(2);
    pq.Push(4);

    std::vector<int> result;
    while (!pq.Empty()) {
        result.push_back(pq.Top());
        pq.Pop();
    }
    EXPECT_EQ(result, (std::vector<int>{5, 4, 3, 2, 1}));
}

// --- Double Type Tests ---

TEST(MinPriorityQueue, DoubleType) {
    MinPriorityQueue<double> pq;
    pq.Push(3.5);
    pq.Push(1.5);
    pq.Push(2.5);
    EXPECT_DOUBLE_EQ(pq.Top(), 1.5);
}
