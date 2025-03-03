/**
 * @file circular_list_test.cpp
 * @brief Unit tests for CircularList (circular_list.hpp).
 *
 * Compile:
 *   g++ -std=c++23 -Wall -Wextra circular_list_test.cpp -lgtest -lgtest_main -pthread -o circular_list_test
 */

#include "CircularList.hpp"

#include <string>
#include <gtest/gtest.h>

// =============================================================================
// Google Test
// =============================================================================

TEST(CircularListTest, DefaultConstruction) {
    CircularList<int> Cl;
    EXPECT_TRUE(Cl.Empty());
    EXPECT_EQ(Cl.Size(), 0u);
}

TEST(CircularListTest, PushBackAndIterate) {
    CircularList<int> Cl;
    Cl.push_back(1);
    Cl.push_back(2);
    Cl.push_back(3);
    EXPECT_EQ(Cl.Front(), 1);
    EXPECT_EQ(Cl.Back(), 3);
    EXPECT_EQ(Cl.Size(), 3u);

    int Expected = 1;
    for (int v : Cl)
        EXPECT_EQ(v, Expected++);
}

TEST(CircularListTest, PushFront) {
    CircularList<int> Cl;
    Cl.push_front(3);
    Cl.push_front(2);
    Cl.push_front(1);
    EXPECT_EQ(Cl.Front(), 1);
    EXPECT_EQ(Cl.Back(), 3);
}

TEST(CircularListTest, PopFrontPopBack) {
    CircularList<int> Cl{1, 2, 3, 4};
    Cl.pop_front();
    EXPECT_EQ(Cl.Front(), 2);
    Cl.pop_back();
    EXPECT_EQ(Cl.Back(), 3);
    EXPECT_EQ(Cl.Size(), 2u);
}

TEST(CircularListTest, EmptyThrows) {
    CircularList<int> Cl;
    EXPECT_THROW(Cl.Front(), std::underflow_error);
    EXPECT_THROW(Cl.Back(), std::underflow_error);
    EXPECT_THROW(Cl.pop_front(), std::underflow_error);
    EXPECT_THROW(Cl.pop_back(), std::underflow_error);
}

TEST(CircularListTest, FindAndRemove) {
    CircularList<int> Cl{10, 20, 30, 40};
    auto It = Cl.Find(30);
    EXPECT_NE(It, Cl.end());
    EXPECT_EQ(*It, 30);

    EXPECT_TRUE(Cl.remove(30));
    EXPECT_EQ(Cl.Find(30), Cl.end());
    EXPECT_EQ(Cl.Size(), 3u);

    EXPECT_FALSE(Cl.remove(99));
}

TEST(CircularListTest, Insert) {
    CircularList<int> Cl{1, 3};
    auto It = Cl.begin();
    ++It;  // points to 3
    Cl.Insert(It, 2);  // insert 2 before 3

    int Expected = 1;
    for (int v : Cl)
        EXPECT_EQ(v, Expected++);
}

TEST(CircularListTest, Erase) {
    CircularList<int> Cl{1, 2, 3};
    auto It = Cl.begin();
    ++It;  // points to 2
    auto NextIt = Cl.Erase(It);
    EXPECT_EQ(*NextIt, 3);
    EXPECT_EQ(Cl.Size(), 2u);
}

TEST(CircularListTest, Clear) {
    CircularList<int> Cl{1, 2, 3};
    Cl.Clear();
    EXPECT_TRUE(Cl.Empty());
    Cl.push_back(42);
    EXPECT_EQ(Cl.Front(), 42);
}

TEST(CircularListTest, InitializerList) {
    CircularList<int> Cl{5, 10, 15};
    EXPECT_EQ(Cl.Size(), 3u);
    EXPECT_EQ(Cl.Front(), 5);
    EXPECT_EQ(Cl.Back(), 15);
}

TEST(CircularListTest, CopyConstructor) {
    CircularList<int> a{1, 2, 3};
    CircularList<int> b(a);
    EXPECT_EQ(b.Size(), 3u);
    b.pop_front();
    EXPECT_EQ(a.Size(), 3u);  // a unaffected
}

TEST(CircularListTest, MoveConstructor) {
    CircularList<int> a{1, 2, 3};
    CircularList<int> b(std::move(a));
    EXPECT_EQ(b.Size(), 3u);
    EXPECT_TRUE(a.Empty());
}

TEST(CircularListTest, Swap) {
    CircularList<int> a{1, 2};
    CircularList<int> b{10, 20, 30};
    a.Swap(b);
    EXPECT_EQ(a.Size(), 3u);
    EXPECT_EQ(a.Front(), 10);
    EXPECT_EQ(b.Size(), 2u);
    EXPECT_EQ(b.Front(), 1);
}

TEST(CircularListTest, ReverseIteration) {
    CircularList<int> Cl{1, 2, 3};
    auto It = Cl.end();
    --It; EXPECT_EQ(*It, 3);
    --It; EXPECT_EQ(*It, 2);
    --It; EXPECT_EQ(*It, 1);
    EXPECT_EQ(It, Cl.begin());
}

TEST(CircularListTest, StringType) {
    CircularList<std::string> Cl;
    Cl.push_back("hello");
    Cl.push_front("world");
    EXPECT_EQ(Cl.Front(), "world");
    EXPECT_EQ(Cl.Back(), "hello");
}
