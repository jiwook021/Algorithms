/**
 * @file main.cpp
 * @brief Driver for TaskPriorityRL
 */

#include "TaskPriorityRL.hpp"
#include <iostream>

 int main() {
     try {
         RunTaskPrioritizationDemo();
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     return 0;
 }