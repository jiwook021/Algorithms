/**
 * @file deque_test.cpp
 * @brief Unit tests for Deque (deque.hpp).
 *
 * Compile:
 *   g++ -std=c++23 -Wall -Wextra deque_test.cpp -lgtest -lgtest_main -pthread -o deque_test
 */

#include "Deque.hpp"

#include <string>
#include <gtest/gtest.h>

// =============================================================================
// Google Test
// =============================================================================

TEST(DequeTest, DefaultConstruction) {
    Deque<int> Dq;
    EXPECT_TRUE(Dq.Empty());
    EXPECT_EQ(Dq.Size(), 0u);
}

TEST(DequeTest, PushFrontPopFront) {
    Deque<int> Dq;
    Dq.PushFront(3);
    Dq.PushFront(2);
    Dq.PushFront(1);
    EXPECT_EQ(Dq.Front(), 1);
    EXPECT_EQ(Dq.Back(), 3);
    EXPECT_EQ(Dq.Size(), 3u);

    Dq.PopFront();
    EXPECT_EQ(Dq.Front(), 2);
}

TEST(DequeTest, PushBackPopBack) {
    Deque<int> Dq;
    Dq.PushBack(1);
    Dq.PushBack(2);
    Dq.PushBack(3);
    EXPECT_EQ(Dq.Back(), 3);

    Dq.PopBack();
    EXPECT_EQ(Dq.Back(), 2);
    EXPECT_EQ(Dq.Size(), 2u);
}

TEST(DequeTest, MixedOperations) {
    Deque<int> Dq;
    Dq.PushBack(10);
    Dq.PushFront(5);
    Dq.PushBack(20);
    // contents: 5, 10, 20
    EXPECT_EQ(Dq.Front(), 5);
    EXPECT_EQ(Dq.Back(), 20);

    Dq.PopFront();
    Dq.PopBack();
    EXPECT_EQ(Dq.Front(), 10);
    EXPECT_EQ(Dq.Back(), 10);
    EXPECT_EQ(Dq.Size(), 1u);
}

TEST(DequeTest, UnderflowThrows) {
    Deque<int> Dq;
    EXPECT_THROW(Dq.PopFront(), std::underflow_error);
    EXPECT_THROW(Dq.PopBack(), std::underflow_error);
    EXPECT_THROW(Dq.Front(), std::underflow_error);
    EXPECT_THROW(Dq.Back(), std::underflow_error);
}

TEST(DequeTest, Clear) {
    Deque<int> Dq{1, 2, 3, 4, 5};
    EXPECT_EQ(Dq.Size(), 5u);
    Dq.Clear();
    EXPECT_TRUE(Dq.Empty());
    Dq.PushBack(42);
    EXPECT_EQ(Dq.Front(), 42);
}

TEST(DequeTest, InitializerList) {
    Deque<int> Dq{10, 20, 30};
    EXPECT_EQ(Dq.Size(), 3u);
    EXPECT_EQ(Dq.Front(), 10);
    EXPECT_EQ(Dq.Back(), 30);
}

TEST(DequeTest, CopyConstructor) {
    Deque<int> a{1, 2, 3};
    Deque<int> b(a);
    EXPECT_EQ(b.Size(), 3u);
    EXPECT_EQ(b.Front(), 1);
    b.PopFront();
    EXPECT_EQ(a.Front(), 1);  // a unaffected
}

TEST(DequeTest, MoveConstructor) {
    Deque<int> a{1, 2, 3};
    Deque<int> b(std::move(a));
    EXPECT_EQ(b.Size(), 3u);
    EXPECT_TRUE(a.Empty());
}

TEST(DequeTest, ForwardIteration) {
    Deque<int> Dq{1, 2, 3, 4};
    int Expected = 1;
    for (auto It = Dq.begin(); It != Dq.end(); ++It) {
        EXPECT_EQ(*It, Expected++);
    }
}

TEST(DequeTest, RangeForLoop) {
    Deque<int> Dq{10, 20, 30};
    int Sum = 0;
    for (int v : Dq) Sum += v;
    EXPECT_EQ(Sum, 60);
}

TEST(DequeTest, EmplaceBack) {
    Deque<std::string> Dq;
    Dq.EmplaceBack(3, 'a');
    EXPECT_EQ(Dq.Back(), "aaa");
}

TEST(DequeTest, Swap) {
    Deque<int> a{1, 2};
    Deque<int> b{10, 20, 30};
    a.Swap(b);
    EXPECT_EQ(a.Size(), 3u);
    EXPECT_EQ(a.Front(), 10);
    EXPECT_EQ(b.Size(), 2u);
    EXPECT_EQ(b.Front(), 1);
}

TEST(DequeTest, StringType) {
    Deque<std::string> Dq;
    Dq.PushBack("hello");
    Dq.PushFront("world");
    EXPECT_EQ(Dq.Front(), "world");
    EXPECT_EQ(Dq.Back(), "hello");
}
