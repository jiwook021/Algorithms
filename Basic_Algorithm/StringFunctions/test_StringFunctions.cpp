/**
 * @file test_StringFunctions.cpp
 * @brief Unit tests for StringFunctions
 */

#include "StringFunctions.hpp"
#include <gtest/gtest.h>
#include <cstring>

using namespace StringFunctions;

// --- MyStrlen ---
TEST(StringFunctions, StrlenEmpty) {
    EXPECT_EQ(MyStrlen(""), 0);
}

TEST(StringFunctions, StrlenSingle) {
    EXPECT_EQ(MyStrlen("a"), 1);
}

TEST(StringFunctions, StrlenNormal) {
    EXPECT_EQ(MyStrlen("Hello"), 5);
}

TEST(StringFunctions, StrlenWithSpaces) {
    EXPECT_EQ(MyStrlen("Hello World"), 11);
}

// --- MyStrcpy ---
TEST(StringFunctions, StrcpyNormal) {
    char dest[20];
    MyStrcpy(dest, "Hello");
    EXPECT_STREQ(dest, "Hello");
}

TEST(StringFunctions, StrcpyEmpty) {
    char dest[20] = "old";
    MyStrcpy(dest, "");
    EXPECT_STREQ(dest, "");
}

TEST(StringFunctions, StrcpyOverwrites) {
    char dest[20] = "LongOldString";
    MyStrcpy(dest, "New");
    EXPECT_STREQ(dest, "New");
}

// --- MyStrrev ---
TEST(StringFunctions, StrrevNormal) {
    char s[] = "abcde";
    MyStrrev(s);
    EXPECT_STREQ(s, "edcba");
}

TEST(StringFunctions, StrrevPalindrome) {
    char s[] = "racecar";
    MyStrrev(s);
    EXPECT_STREQ(s, "racecar");
}

TEST(StringFunctions, StrrevSingle) {
    char s[] = "x";
    MyStrrev(s);
    EXPECT_STREQ(s, "x");
}

TEST(StringFunctions, StrrevEmpty) {
    char s[] = "";
    MyStrrev(s);
    EXPECT_STREQ(s, "");
}

TEST(StringFunctions, StrrevEvenLength) {
    char s[] = "abcd";
    MyStrrev(s);
    EXPECT_STREQ(s, "dcba");
}

// --- MyStrcmp ---
TEST(StringFunctions, StrcmpEqual) {
    EXPECT_EQ(MyStrcmp("abc", "abc"), 0);
}

TEST(StringFunctions, StrcmpLessThan) {
    EXPECT_LT(MyStrcmp("abc", "abd"), 0);
}

TEST(StringFunctions, StrcmpGreaterThan) {
    EXPECT_GT(MyStrcmp("abd", "abc"), 0);
}

TEST(StringFunctions, StrcmpPrefixShorter) {
    EXPECT_LT(MyStrcmp("ab", "abc"), 0);
}

TEST(StringFunctions, StrcmpPrefixLonger) {
    EXPECT_GT(MyStrcmp("abc", "ab"), 0);
}

TEST(StringFunctions, StrcmpBothEmpty) {
    EXPECT_EQ(MyStrcmp("", ""), 0);
}

// --- MyStrcat ---
TEST(StringFunctions, StrcatNormal) {
    char dest[50] = "Hello";
    MyStrcat(dest, " World");
    EXPECT_STREQ(dest, "Hello World");
}

TEST(StringFunctions, StrcatEmptySrc) {
    char dest[50] = "Hello";
    MyStrcat(dest, "");
    EXPECT_STREQ(dest, "Hello");
}

TEST(StringFunctions, StrcatEmptyDest) {
    char dest[50] = "";
    MyStrcat(dest, "World");
    EXPECT_STREQ(dest, "World");
}

TEST(StringFunctions, StrcatBothEmpty) {
    char dest[50] = "";
    MyStrcat(dest, "");
    EXPECT_STREQ(dest, "");
}
