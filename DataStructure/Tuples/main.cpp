/**
 * @file main.cpp
 * @brief Driver for Tuples utilities
 */

#include "Tuples.hpp"
#include <iostream>

int main() {
    auto studentDesc = std::make_tuple("ID", "Name", "GPA");
    auto student = std::make_tuple(123456, "John Doe", 3.7);

    std::cout << studentDesc << "\n"
              << student << "\n";

    std::cout << std::tuple_cat(studentDesc, student) << "\n";

    auto zipped = Zip(studentDesc, student);
    std::cout << zipped << "\n";

    auto numbers = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::cout << Zip(
                    std::make_tuple("Sum", "Minimum", "Maximum", "Average"),
                    SumMinMaxAvg(numbers))
              << "\n";

    return 0;
}
