# Code Overview: processing.cpp

### Purpose of the Code

The purpose of this code is to provide functionality for rotating images either 90 degrees to the left (counterclockwise) or 90 degrees to the right (clockwise). This is a common operation in image processing, often used in applications like photo editing, computer vision, and graphics rendering.

### Main Functionality

The code defines two main functions:
1. **`rotate_left(Image* img)`**: Rotates the given image 90 degrees counterclockwise.
2. **`rotate_right(Image* img)`**: Rotates the given image 90 degrees clockwise.

These functions manipulate an `Image` object, which is presumably a custom data structure defined in the `Image.h` header file. The functions use an auxiliary (temporary) image to store the rotated image before copying it back to the original image.

### Algorithms Used

The core algorithm used in both functions is **pixel mapping**. When rotating an image, each pixel in the original image is mapped to a new position in the rotated image. The mapping depends on the direction of rotation:

- **Rotate Left (Counterclockwise)**:
  - The pixel at position `(r, c)` in the original image is moved to position `(width - 1 - c, r)` in the rotated image.
  - This effectively swaps the rows and columns and adjusts the indices to account for the rotation.

- **Rotate Right (Clockwise)**:
  - The pixel at position `(r, c)` in the original image is moved to position `(c, height - 1 - r)` in the rotated image.
  - This also swaps the rows and columns but adjusts the indices differently to achieve a clockwise rotation.

### Overall Structure

The code is structured as follows:

1. **Includes and Namespace**:
   - The code includes necessary headers (`<cassert>`, `"processing.h"`, `"Image.h"`, `"Matrix.h"`, etc.) and uses the `std` namespace for standard library functions.

2. **Rotate Left Function**:
   - The function takes a pointer to an `Image` object as input.
   - It calculates the width and height of the image.
   - It creates an auxiliary image (`aux`) with swapped dimensions (height becomes width and vice versa).
   - It iterates over each pixel in the original image and maps it to the correct position in the auxiliary image.
   - Finally, it copies the rotated image back to the original image and deletes the auxiliary image to free memory.

3. **Rotate Right Function**:
   - The function is similar to `rotate_left`, but the pixel mapping is adjusted to achieve a clockwise rotation.
   - The auxiliary image is created with swapped dimensions, and pixels are mapped accordingly.
   - The rotated image is copied back to the original image, and the auxiliary image is deleted.

### Problem Being Solved

The problem being solved is the need to rotate images efficiently while preserving their pixel data. This is a fundamental operation in image processing, and the code provides a straightforward implementation that handles the rotation by mapping each pixel to its new position.

### Approach Taken

The approach taken is to:
1. **Create a Temporary Image**: An auxiliary image is created to store the rotated image temporarily. This avoids modifying the original image directly, which could lead to incorrect results if the rotation is done in-place.
2. **Pixel Mapping**: Each pixel in the original image is mapped to its new position in the auxiliary image based on the rotation direction.
3. **Copy Back**: After the rotation is complete, the rotated image is copied back to the original image, effectively updating it with the rotated version.
4. **Memory Management**: The auxiliary image is deleted to prevent memory leaks.

### How the Different Parts of the Code Work Together

- **Image Manipulation Functions**: The code relies on functions like `Image_width`, `Image_height`, `Image_init`, `Image_set_pixel`, and `Image_get_pixel`, which are presumably defined in the `Image.h` header. These functions handle the low-level details of image manipulation.
- **Auxiliary Image**: The auxiliary image is used as a temporary buffer to store the rotated image. This ensures that the original image is not modified until the rotation is complete.
- **Pixel Mapping**: The nested loops iterate over each pixel in the original image and map it to the correct position in the auxiliary image. This is the core of the rotation algorithm.
- **Memory Management**: The use of `new` and `delete` ensures that the auxiliary image is properly allocated and deallocated, preventing memory leaks.

### Summary

In summary, this code provides a clear and efficient implementation of image rotation. It uses pixel mapping to rotate images either 90 degrees counterclockwise or clockwise, and it handles memory management carefully to avoid leaks. The code is well-structured and relies on helper functions to abstract away the details of image manipulation, making it easier to understand and maintain.