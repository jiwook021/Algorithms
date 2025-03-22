# Step-by-Step Explanation: processing.cpp

### Comprehensive, Step-by-Step Explanation of the Code

Let’s break down the code line by line, explaining every detail in a way that’s accessible to beginners while still being thorough for advanced learners. We’ll focus on the `rotate_left` function since it’s fully implemented, and the `rotate_right` function follows a similar pattern.

---

### **1. Includes and Namespace**
```cpp
#include <cassert>
#include "processing.h"
#include "Image.h"
#include "Matrix.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cassert>

using namespace std;
```

#### What It Does:
- **`#include` Statements**: These bring in external code libraries or files that the program needs to run.
  - `<cassert>`: Provides debugging tools to check if certain conditions are true (used for error handling).
  - `"processing.h"`, `"Image.h"`, `"Matrix.h"`: These are custom header files that define functions and data structures used in the program.
  - `<iostream>`, `<string>`, `<sstream>`: Standard C++ libraries for input/output, string manipulation, and string streams.

- **`using namespace std;`**: This allows us to use standard library functions (like `cout` or `string`) without typing `std::` every time.

#### Why It’s Used:
- Including the right headers ensures the program has access to the functions and data structures it needs.
- `using namespace std;` is a convenience to make the code shorter and easier to read.

---

### **2. The `rotate_left` Function**
```cpp
void rotate_left(Image* img) {
```

#### What It Does:
- This function rotates an image 90 degrees counterclockwise.
- It takes a pointer to an `Image` object (`Image* img`) as input. A **pointer** is a variable that stores the memory address of another variable (in this case, the image).

#### Why It’s Used:
- Passing a pointer to the image allows the function to modify the original image directly, rather than working with a copy.

---

### **3. Extracting Image Dimensions**
```cpp
int width = Image_width(img);
int height = Image_height(img);
```

#### What It Does:
- These lines get the width and height of the image using the `Image_width` and `Image_height` functions.
- The results are stored in the variables `width` and `height`.

#### Why It’s Used:
- The dimensions of the image are needed to create a new image (the auxiliary image) with swapped dimensions (height becomes width, and vice versa).

---

### **4. Creating an Auxiliary Image**
```cpp
Image *aux = new Image;
Image_init(aux, height, width);
```

#### What It Does:
- **`Image *aux = new Image;`**: This creates a new `Image` object dynamically (in the heap) and stores its address in the pointer `aux`.
- **`Image_init(aux, height, width);`**: This initializes the auxiliary image with swapped dimensions. For example, if the original image is 100x200 (width x height), the auxiliary image will be 200x100.

#### Why It’s Used:
- The auxiliary image is needed to store the rotated version of the original image temporarily. This avoids overwriting the original image while the rotation is in progress.

---

### **5. Pixel Mapping (The Core Algorithm)**
```cpp
for (int r = 0; r < height; ++r) {
  for (int c = 0; c < width; ++c) {
    Image_set_pixel(aux, width - 1 - c, r, Image_get_pixel(img, r, c));
  }
}
```

#### What It Does:
- This is a **nested loop** that iterates over every pixel in the original image.
  - The outer loop (`for (int r = 0; r < height; ++r)`) iterates over the rows (height) of the image.
  - The inner loop (`for (int c = 0; c < width; ++c)`) iterates over the columns (width) of the image.
- **`Image_get_pixel(img, r, c)`**: This retrieves the pixel at position `(r, c)` in the original image.
- **`Image_set_pixel(aux, width - 1 - c, r, ...)`**: This sets the pixel in the auxiliary image at position `(width - 1 - c, r)` to the value of the pixel from the original image.

#### Why It’s Used:
- This is the core of the rotation algorithm. It maps each pixel from the original image to its new position in the rotated image.
- The formula `(width - 1 - c, r)` ensures that the image is rotated 90 degrees counterclockwise.

#### Example:
Suppose the original image is a 3x3 grid:
```
A B C
D E F
G H I
```
After rotation, the auxiliary image will look like this:
```
C F I
B E H
A D G
```
- Pixel `A` (0,0) moves to (2,0).
- Pixel `B` (0,1) moves to (1,0).
- Pixel `C` (0,2) moves to (0,0).
- And so on.

---

### **6. Copying the Rotated Image Back**
```cpp
*img = *aux;
```

#### What It Does:
- This line copies the contents of the auxiliary image (`aux`) back into the original image (`img`).
- The `*` operator is used to dereference the pointers, accessing the actual `Image` objects they point to.

#### Why It’s Used:
- After the rotation is complete, the original image needs to be updated with the rotated version.

---

### **7. Cleaning Up Memory**
```cpp
delete aux;
```

#### What It Does:
- This deletes the auxiliary image from memory, freeing up the space it occupied.

#### Why It’s Used:
- In C++, dynamically allocated memory (created with `new`) must be manually freed (with `delete`) to avoid memory leaks.

---

### **8. The `rotate_right` Function**
```cpp
void rotate_right(Image* img) {
  // Similar to rotate_left, but with different pixel mapping
}
```

#### What It Does:
- This function is similar to `rotate_left`, but it rotates the image 90 degrees clockwise instead of counterclockwise.
- The pixel mapping formula would be different to achieve the clockwise rotation.

#### Why It’s Used:
- Clockwise rotation is another common operation, and having a separate function for it makes the code modular and reusable.

---

### **Summary of the Code’s Flow**
1. **Input**: The function receives a pointer to an image.
2. **Setup**: It extracts the image’s dimensions and creates an auxiliary image with swapped dimensions.
3. **Rotation**: It maps each pixel from the original image to its new position in the auxiliary image.
4. **Update**: It copies the rotated image back to the original image.
5. **Cleanup**: It deletes the auxiliary image to free memory.

---

### **Key Concepts Explained**
- **Pointer**: A variable that stores the memory address of another variable. Used here to modify the original image directly.
- **Dynamic Memory Allocation**: Using `new` to create objects in the heap (a region of memory) and `delete` to free them.
- **Pixel Mapping**: The process of determining where each pixel should go in the rotated image.
- **Nested Loops**: A loop inside another loop, used here to iterate over every pixel in the image.

---

### **Text-Based Diagram of Pixel Mapping**
Original Image (3x3):
```
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)
(2,0) (2,1) (2,2)
```

After Rotation (Counterclockwise):
```
(0,2) (1,2) (2,2)
(0,1) (1,1) (2,1)
(0,0) (1,0) (2,0)
```

This diagram shows how the pixel at `(r, c)` in the original image moves to `(width - 1 - c, r)` in the rotated image.

---

By breaking down the code step by step, we’ve made it accessible to beginners while still covering all the technical details. Let me know if you’d like further clarification!