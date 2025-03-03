/**
 * @file StringFunctions.hpp
 * @brief Custom C string function implementations
 * @details Hand-rolled versions of strlen, strcpy, strrev, strcmp, and strcat
 *          without using any standard library string functions.
 */

#pragma once

namespace StringFunctions {

/**
 * @brief Compute string length (excluding null terminator).
 */
int MyStrlen(const char* str);

/**
 * @brief Copy src into dest (including null terminator).
 */
void MyStrcpy(char* dest, const char* src);

/**
 * @brief Reverse a string in-place using two-pointer swap.
 */
void MyStrrev(char* str);

/**
 * @brief Lexicographic comparison of two strings.
 * @return <0 if str1 < str2, 0 if equal, >0 if str1 > str2
 */
int MyStrcmp(const char* str1, const char* str2);

/**
 * @brief Append src to the end of dest.
 */
void MyStrcat(char* dest, const char* src);

} // namespace StringFunctions
