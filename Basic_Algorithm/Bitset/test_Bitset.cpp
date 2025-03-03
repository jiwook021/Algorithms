/**
 * @file test_Bitset.cpp
 * @brief Unit tests for Bitset
 */

#include "Bitset.hpp"
#include <gtest/gtest.h>

TEST(Bitset, SetAndTest) {
    MyBitset<8> b;
    b.Set(3);
    EXPECT_TRUE(b.Test(3));
    EXPECT_FALSE(b.Test(0));
}

TEST(Bitset, Reset) {
    MyBitset<8> b;
    b.Set(3);
    b.Reset(3);
    EXPECT_FALSE(b.Test(3));
}

TEST(Bitset, Flip) {
    MyBitset<8> b;
    b.Flip(0);
    EXPECT_TRUE(b.Test(0));
    b.Flip(0);
    EXPECT_FALSE(b.Test(0));
}

TEST(Bitset, SetAll) {
    MyBitset<8> b;
    b.SetAll();
    EXPECT_TRUE(b.All());
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(b.Test(i));
    }
}

TEST(Bitset, ResetAll) {
    MyBitset<8> b;
    b.SetAll();
    b.ResetAll();
    EXPECT_TRUE(b.None());
}

TEST(Bitset, FlipAll) {
    MyBitset<8> b;
    b.Set(0);
    b.Set(2);
    b.FlipAll();
    EXPECT_FALSE(b.Test(0));
    EXPECT_TRUE(b.Test(1));
    EXPECT_FALSE(b.Test(2));
    EXPECT_TRUE(b.Test(3));
}

TEST(Bitset, NoneOnEmpty) {
    MyBitset<8> b;
    EXPECT_TRUE(b.None());
    b.Set(0);
    EXPECT_FALSE(b.None());
}

TEST(Bitset, AnyDetectsBit) {
    MyBitset<8> b;
    EXPECT_FALSE(b.Any());
    b.Set(5);
    EXPECT_TRUE(b.Any());
}

TEST(Bitset, OutOfRangeThrows) {
    MyBitset<8> b;
    EXPECT_THROW(b.Set(8), std::out_of_range);
    EXPECT_THROW(b.Reset(8), std::out_of_range);
    EXPECT_THROW(b.Flip(8), std::out_of_range);
    EXPECT_THROW(b.Test(8), std::out_of_range);
}

TEST(Bitset, ToString) {
    MyBitset<4> b;
    b.Set(0);
    b.Set(2);
    EXPECT_EQ(b.ToString(), "0101");
}

TEST(Bitset, SmallBitset) {
    MyBitset<1> b;
    EXPECT_TRUE(b.None());
    b.Set(0);
    EXPECT_TRUE(b.All());
    EXPECT_EQ(b.ToString(), "1");
}
