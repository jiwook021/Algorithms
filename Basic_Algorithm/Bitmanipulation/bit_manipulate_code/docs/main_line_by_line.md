# Step-by-Step Explanation: main.c

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple terms, examples, and diagrams to make everything clear, even for someone who is just starting to learn programming.

---

### **1. `#include` Statements**
```c
#include <stdio.h>
#include <stdbool.h>
```
- **What it does**: These lines include standard libraries in the program.
  - `stdio.h` provides functions like `printf` for printing to the console.
  - `stdbool.h` allows us to use the `bool` type (true/false values) in C.
- **Why it’s used**: Libraries provide pre-written code for common tasks, so we don’t have to reinvent the wheel.

---

### **2. `reversebit` Function**
```c
char reversebit(char input) {
    char output = 0;
    int sz = sizeof(input) * 8;
    for (int i = sz; i > 0; i--) {
        if (input & 1 << i) {
            output |= 1 << (sz - 1 - i);
        }
    }
    return output;
}
```

#### **What it does**:
This function reverses the bits of a `char` (8 bits). For example:
- Input: `00001100` (12 in decimal)
- Output: `00110000` (48 in decimal)

#### **Step-by-Step Explanation**:
1. **`char output = 0;`**:
   - Initializes a variable `output` to store the reversed bits. It starts as `00000000`.

2. **`int sz = sizeof(input) * 8;`**:
   - `sizeof(input)` gives the size of `input` in bytes. For a `char`, this is 1 byte (8 bits).
   - `sz` is set to 8, the number of bits in a `char`.

3. **`for (int i = sz; i > 0; i--)`**:
   - A loop that iterates over each bit position, starting from the most significant bit (MSB) to the least significant bit (LSB).

4. **`if (input & 1 << i)`**:
   - Checks if the bit at position `i` in `input` is set (1).
   - `1 << i` creates a mask with a `1` at position `i`. For example, if `i = 3`, the mask is `00001000`.
   - `input & mask` checks if the bit at position `i` in `input` is `1`.

5. **`output |= 1 << (sz - 1 - i);`**:
   - If the bit is set, it sets the corresponding bit in `output`.
   - `sz - 1 - i` calculates the mirrored position for the reversed bit. For example, if `i = 3`, the mirrored position is `4`.

6. **`return output;`**:
   - Returns the reversed bit pattern.

#### **Example**:
- Input: `00001100` (12)
- Loop:
  - Bit 3 is set (`00001000`), so set bit 4 in `output`.
  - Bit 2 is set (`00000100`), so set bit 5 in `output`.
- Output: `00110000` (48)

---

### **3. `bitExtracted` Function**
```c
int bitExtracted(int number, int k, int s) {
    return (((1 << k) - 1) & (number >> (s - 1)));
}
```

#### **What it does**:
Extracts `k` bits starting from position `s` in `number`. For example:
- Input: `number = 108` (`01101100`), `k = 4`, `s = 3`
- Output: `0110` (6 in decimal)

#### **Step-by-Step Explanation**:
1. **`(1 << k) - 1`**:
   - Creates a mask with `k` bits set to `1`. For `k = 4`, the mask is `00001111`.

2. **`number >> (s - 1)`**:
   - Shifts `number` right by `s - 1` positions to align the desired bits with the mask. For `s = 3`, `number` becomes `00011011`.

3. **`&` (AND operation)**:
   - Applies the mask to extract the desired bits. For example:
     ```
     00011011 (shifted number)
     & 00001111 (mask)
     ---------
     00001011 (extracted bits)
     ```

4. **Return the result**.

---

### **4. `swapBits` Function**
```c
unsigned int swapBits(unsigned int n, int p1, int p2) {
    unsigned int bit1 = (n >> p1) & 1;
    unsigned int bit2 = (n >> p2) & 1;
    unsigned int x = (bit1 ^ bit2);
    x = (x << p1) | (x << p2);
    unsigned int result = n ^ x;
    return result;
}
```

#### **What it does**:
Swaps two bits at positions `p1` and `p2` in `n`. For example:
- Input: `n = 28` (`11100`), `p1 = 0`, `p2 = 3`
- Output: `21` (`10101`)

#### **Step-by-Step Explanation**:
1. **Extract bits**:
   - `bit1 = (n >> p1) & 1`: Extracts the bit at `p1`.
   - `bit2 = (n >> p2) & 1`: Extracts the bit at `p2`.

2. **XOR the bits**:
   - `x = (bit1 ^ bit2)`: If the bits are different, `x` is `1`; otherwise, `0`.

3. **Shift and combine**:
   - `x = (x << p1) | (x << p2)`: Places `x` at both `p1` and `p2`.

4. **Swap the bits**:
   - `result = n ^ x`: XORs `n` with `x` to swap the bits.

---

### **5. `reverseDigits` Function**
```c
int reverseDigits(int num) {
    int reversedNum = 0;
    while (num != 0) {
        reversedNum = reversedNum * 10 + num % 10;
        num /= 10;
    }
    return reversedNum;
}
```

#### **What it does**:
Reverses the digits of `num`. For example:
- Input: `123`
- Output: `321`

#### **Step-by-Step Explanation**:
1. **Initialize `reversedNum` to 0**.
2. **Loop**:
   - Extract the last digit of `num` using `num % 10`.
   - Append it to `reversedNum` by multiplying by 10 and adding the digit.
   - Remove the last digit from `num` using `num /= 10`.
3. **Return `reversedNum`**.

---

### **6. `intersect` Function**
```c
bool intersect(Rectangle r1, Rectangle r2, Rectangle *intersection) {
    int xOverlap = min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x);
    int yOverlap = min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y);

    if (xOverlap > 0 && yOverlap > 0) {
        intersection->x = max(r1.x, r2.x);
        intersection->y = max(r1.y, r2.y);
        intersection->width = xOverlap;
        intersection->height = yOverlap;
        return true;
    }
    return false;
}
```

#### **What it does**:
Checks if two rectangles intersect and calculates the intersection area.

#### **Step-by-Step Explanation**:
1. **Calculate overlap**:
   - `xOverlap`: The horizontal overlap between the rectangles.
   - `yOverlap`: The vertical overlap between the rectangles.
2. **Check for intersection**:
   - If both `xOverlap` and `yOverlap` are positive, the rectangles intersect.
3. **Calculate intersection rectangle**:
   - Set the `x`, `y`, `width`, and `height` of the intersection.
4. **Return `true` if they intersect, otherwise `false`**.

---

### **7. `main` Function**
```c
int main() {
    // printf("%d\n", reversebit(12)); // Example usage
    // printf("%d\n", bitExtracted(108,4,3)); // Example usage
    // swap();
    // reversedigit();
}
```

#### **What it does**:
The `main` function is the entry point of the program. It is currently commented out but can be used to test the other functions.

---

### **Summary**
This code is a collection of utility functions that perform various tasks, such as bit manipulation, number reversal, and geometric calculations. Each function is modular and can be used independently. The code is well-structured and demonstrates fundamental programming concepts like loops, conditionals, and bitwise operations.