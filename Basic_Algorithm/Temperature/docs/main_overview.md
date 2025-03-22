# Code Overview: main.c

### Purpose of the Code

This C program is a **temperature conversion tool** that converts temperatures between **Celsius** and **Fahrenheit**. It takes input from the user in the form of a temperature value and its corresponding unit (either `C` for Celsius or `F` for Fahrenheit). Based on the unit provided, the program converts the temperature to the other unit and displays the result.

#### Problem Being Solved
The problem being solved is the need to convert temperatures between two commonly used temperature scales: Celsius and Fahrenheit. This is a common task in scientific, engineering, and everyday contexts, especially when dealing with international data or weather reports.

#### Approach Taken
The program uses **mathematical formulas** to perform the conversions:
1. **Celsius to Fahrenheit**: The formula used is:
   \[
   F = \left(\frac{9}{5} \times C\right) + 32
   \]
   This is implemented in the `ctof` function.

2. **Fahrenheit to Celsius**: The formula used is:
   \[
   C = \frac{5}{9} \times (F - 32)
   \]
   This is implemented in the `ftoc` function.

The program is structured to:
1. Take user input for the temperature and its unit.
2. Use conditional logic to determine which conversion to perform.
3. Display the converted temperature with one decimal place.

#### Overall Structure
The code is divided into three main parts:
1. **Main Function (`main`)**:
   - Handles user input and output.
   - Uses conditional statements to decide which conversion function to call.
2. **Conversion Functions (`ctof` and `ftoc`)**:
   - Perform the mathematical calculations for temperature conversion.
3. **Standard Library Inclusion (`#include <stdio.h>`)**:
   - Provides input/output functionality (e.g., `scanf` and `printf`).

#### How the Parts Work Together
1. The program starts in the `main` function, where it prompts the user for input.
2. The user provides a temperature value and its unit (`C` or `F`).
3. The program checks the unit using an `if-else` statement:
   - If the unit is `C`, it calls the `ctof` function to convert Celsius to Fahrenheit.
   - If the unit is `F`, it calls the `ftoc` function to convert Fahrenheit to Celsius.
4. The converted temperature is displayed using `printf`.
5. The program ends by returning `0`, indicating successful execution.

#### Algorithms Used
The algorithms used are straightforward mathematical formulas:
1. **Celsius to Fahrenheit**:
   - Multiply the Celsius value by \( \frac{9}{5} \).
   - Add 32 to the result.
2. **Fahrenheit to Celsius**:
   - Subtract 32 from the Fahrenheit value.
   - Multiply the result by \( \frac{5}{9} \).

These formulas are implemented in the `ctof` and `ftoc` functions, respectively.

#### Example
If the user inputs `25 C`, the program will:
1. Recognize the unit as `C`.
2. Call the `ctof` function with the value `25`.
3. Calculate \( \left(\frac{9}{5} \times 25\right) + 32 = 77 \).
4. Display `77.0 F`.

If the user inputs `77 F`, the program will:
1. Recognize the unit as `F`.
2. Call the `ftoc` function with the value `77`.
3. Calculate \( \frac{5}{9} \times (77 - 32) = 25 \).
4. Display `25.0 C`.

This program is a simple yet effective tool for temperature conversion, demonstrating the use of functions, conditional logic, and basic input/output in C.