/**
 * @file main.cpp
 * @brief Driver for StringFunctions
 */

#include "StringFunctions.hpp"
#include <cstdio>

using namespace StringFunctions;

int main() {
    char str1[50] = "Hello";
    char str2[50] = "World";

    printf("Length of str1: %d\n", MyStrlen(str1));

    MyStrcpy(str1, str2);
    printf("str1 after copy: %s\n", str1);

    MyStrrev(str1);
    printf("str1 after reverse: %s\n", str1);

    printf("strcmp(\"abc\", \"abc\") = %d\n", MyStrcmp("abc", "abc"));
    printf("strcmp(\"abc\", \"abd\") = %d\n", MyStrcmp("abc", "abd"));

    char str3[50] = "Hello";
    MyStrcat(str3, " World");
    printf("str3 after concat: %s\n", str3);

    return 0;
}
