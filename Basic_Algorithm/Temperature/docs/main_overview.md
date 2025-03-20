# Code Overview: main.c

### Purpose of the Code

This C program is a **temperature conversion tool** that converts temperatures between **Celsius** and **Fahrenheit**. It takes input from the user in the form of a temperature value and its corresponding unit (either `C` for Celsius or `F` for Fahrenheit). Based on the unit provided, the program converts the temperature to the other unit and displays the result.

#### Problem Being Solved
The problem being solved is the need to convert temperatures between two commonly used temperature scales: Celsius and Fahrenheit. This is a common task in science, engineering, and everyday life, especially when dealing with international data or recipes.

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
1. Take user input (temperature value and unit).
2. Determine which conversion to perform based on the unit.
3. Call the appropriate conversion function.
4. Display the converted temperature.

#### Overall Structure
The code is divided into three main parts:
1. **Main Function (`main`)**:
   - Handles user input and output.
   - Decides which conversion function to call based on the unit provided by the user.

2. **Conversion Functions (`ctof` and `ftoc`)**:
   - Perform the actual temperature conversions using the formulas mentioned above.

3. **Standard Library Inclusion (`#include <stdio.h>`)**:
   - Provides functions like `scanf` and `printf` for input and output operations.

#### How the Parts Work Together
1. The program starts in the `main` function.
2. It prompts the user to input a temperature and its unit (though the prompt is implicit in the `scanf` function).
3. Based on the unit (`C` or `F`), the program calls either `ctof` or `ftoc` to perform the conversion.
4. The converted temperature is then displayed using `printf`.

### Algorithms Used
The algorithms used are straightforward mathematical formulas:
1. **Celsius to Fahrenheit**:
   - Multiply the Celsius value by \( \frac{9}{5} \).
   - Add 32 to the result.

2. **Fahrenheit to Celsius**:
   - Subtract 32 from the Fahrenheit value.
   - Multiply the result by \( \frac{5}{9} \).

These formulas are implemented in the `ctof` and `ftoc` functions, respectively.

### Key Features
- **User Input Handling**: The program uses `scanf` to read the temperature and unit from the user.
- **Conditional Logic**: The `if-else` statement determines which conversion to perform.
- **Modular Design**: The conversion logic is separated into two functions (`ctof` and `ftoc`), making the code easier to read, maintain, and reuse.
- **Precision Control**: The `printf` function uses `%.1f` to display the converted temperature with one decimal place, ensuring readability.

### Example Usage
If the user inputs:
```
25 C
```
The program will output:
```
77.0 F
```
If the user inputs:
```
77 F
```
The program will output:
```
25.0 C
```

This program is a simple yet effective tool for temperature conversion, demonstrating basic input/output, conditional logic, and modular programming in C.