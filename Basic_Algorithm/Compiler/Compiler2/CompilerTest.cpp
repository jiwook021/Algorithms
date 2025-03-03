/**
 * test.cpp
 * Test program for the SimpleCppCompiler
 * 
 * This program demonstrates various features of C++ that the compiler should handle:
 * - Variable declarations and initializations
 * - Arithmetic and logical operations
 * - Control flow statements (if, while, for)
 * - Function declarations and calls
 * - Basic standard library functions
 */

 #include <iostream>

 // Function declarations
 int Factorial(int n);
 int Fibonacci(int n);
 int Gcd(int a, int b);
 bool IsPrime(int n);
 void PrintArray(int Arr[], int size);
 void BubbleSort(int Arr[], int size);
 
 // Global variable
 const int MAX_SIZE = 10;
 
 /**
  * Main function
  */
 int main() {
     // Print welcome message
     std::cout << "Test program for SimpleCppCompiler" << std::endl;
     
     // Variable declarations and initializations
     int x = 5;
     int y = 10;
     int Sum = x + y;
     int Product = x * y;
     
     // Print variables
     std::cout << "x = " << x << std::endl;
     std::cout << "y = " << y << std::endl;
     std::cout << "sum = " << Sum << std::endl;
     std::cout << "product = " << Product << std::endl;
     
     // If statement
     if (x < y) {
         std::cout << "x is less than y" << std::endl;
     } else {
         std::cout << "x is greater than or equal to y" << std::endl;
     }
     
     // For loop
     std::cout << "Counting from 1 to 5:" << std::endl;
     for (int i = 1; i <= 5; i++) {
         std::cout << i << " ";
     }
     std::cout << std::endl;
     
     // While loop
     std::cout << "Powers of 2 less than 100:" << std::endl;
     int Power = 1;
     while (Power < 100) {
         std::cout << Power << " ";
         Power *= 2;
     }
     std::cout << std::endl;
     
     // Function calls
     std::cout << "Factorial of 5 = " << Factorial(5) << std::endl;
     std::cout << "Fibonacci of 10 = " << Fibonacci(10) << std::endl;
     std::cout << "GCD of 24 and 36 = " << Gcd(24, 36) << std::endl;
     std::cout << "Is 17 prime? " << (IsPrime(17) ? "Yes" : "No") << std::endl;
     
     // Array operations
     int Arr[MAX_SIZE] = {9, 7, 5, 3, 1, 2, 4, 6, 8, 0};
     
     std::cout << "Original array: ";
     PrintArray(Arr, MAX_SIZE);
     
     BubbleSort(Arr, MAX_SIZE);
     
     std::cout << "Sorted array: ";
     PrintArray(Arr, MAX_SIZE);
     
     return 0;
 }
 
 /**
  * Calculate the factorial of a number
  * @param n The number
  * @return The factorial of n
  */
 int Factorial(int n) {
     if (n <= 1) {
         return 1;
     } else {
         return n * Factorial(n - 1);
     }
 }
 
 /**
  * Calculate the nth Fibonacci number
  * @param n The index
  * @return The nth Fibonacci number
  */
 int Fibonacci(int n) {
     if (n <= 0) {
         return 0;
     } else if (n == 1) {
         return 1;
     } else {
         return Fibonacci(n - 1) + Fibonacci(n - 2);
     }
 }
 
 /**
  * Calculate the greatest common divisor of two numbers
  * @param a The first number
  * @param b The second number
  * @return The GCD of a and b
  */
 int Gcd(int a, int b) {
     if (b == 0) {
         return a;
     } else {
         return Gcd(b, a % b);
     }
 }
 
 /**
  * Check if a number is prime
  * @param n The number to check
  * @return True if n is prime, false otherwise
  */
 bool IsPrime(int n) {
     if (n <= 1) {
         return false;
     }
     
     for (int i = 2; i * i <= n; i++) {
         if (n % i == 0) {
             return false;
         }
     }
     
     return true;
 }
 
 /**
  * Print an array
  * @param arr The array to print
  * @param size The size of the array
  */
 void PrintArray(int Arr[], int size) {
     for (int i = 0; i < size; i++) {
         std::cout << Arr[i] << " ";
     }
     std::cout << std::endl;
 }
 
 /**
  * Sort an array using bubble sort
  * @param arr The array to sort
  * @param size The size of the array
  */
 void BubbleSort(int Arr[], int size) {
     for (int i = 0; i < size - 1; i++) {
         for (int j = 0; j < size - i - 1; j++) {
             if (Arr[j] > Arr[j + 1]) {
                 // Swap arr[j] and arr[j+1]
                 int Temp = Arr[j];
                 Arr[j] = Arr[j + 1];
                 Arr[j + 1] = Temp;
             }
         }
     }
 }