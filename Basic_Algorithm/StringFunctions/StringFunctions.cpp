/**
 * @file StringFunctions.cpp
 * @brief Implementation of custom string functions
 */

#include "StringFunctions.hpp"

namespace StringFunctions {

int MyStrlen(const char* str) {
    int i = 0;
    while (str[i] != '\0') {
        ++i;
    }
    return i;
}

void MyStrcpy(char* dest, const char* src) {
    while (*src != '\0') {
        *dest++ = *src++;
    }
    *dest = '\0';
}

void MyStrrev(char* str) {
    int len = MyStrlen(str);
    for (int i = 0; i < len / 2; ++i) {
        char tmp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = tmp;
    }
}

int MyStrcmp(const char* str1, const char* str2) {
    int i = 0;
    while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i]) {
        ++i;
    }
    return str1[i] - str2[i];
}

void MyStrcat(char* dest, const char* src) {
    int len1 = MyStrlen(dest);
    int len2 = MyStrlen(src);
    for (int i = len1; i <= len1 + len2; ++i) {
        dest[i] = src[i - len1];
    }
}

} // namespace StringFunctions
