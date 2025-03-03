/**
 * @file main.cpp
 * @brief Driver for MergeSortThread
 */

#include "MergeSortThread.hpp"

using namespace MergeSortThread;

int main() {
    std::cout << "Hardware threads: "
              << std::thread::hardware_concurrency() << std::endl;

    // Sort integers ascending
    std::vector<int> numbers = {9, 3, 7, 1, 5, 8, 2, 4, 6};
    PrintVector(numbers, "Before");
    MySort(numbers.begin(), numbers.end());
    PrintVector(numbers, "After");

    // Sort descending
    std::vector<int> desc = {9, 3, 7, 1, 5, 8, 2, 4, 6};
    MySort(desc.begin(), desc.end(), [](int a, int b) { return a > b; });
    PrintVector(desc, "Descending");

    // Sort structs
    std::vector<Person> people = {
        {"Alice", 30}, {"Bob", 25}, {"Charlie", 35}, {"David", 20}
    };
    MySort(people.begin(), people.end(),
           [](const Person& a, const Person& b) { return a.Age < b.Age; });
    PrintVector(people, "By age");

    return 0;
}
