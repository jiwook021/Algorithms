# Code Overview: main.cpp

This C++ code is designed to **load an image, apply a Gaussian blur to it, and save the processed image**. Let's break down the purpose, functionality, and structure of the code in detail:

---

### **Purpose of the Code**
The code is an **image processing tool** that performs the following tasks:
1. **Loads an image** from a file (e.g., `input.jpg`).
2. **Applies a Gaussian blur** to the image, which is a common image processing technique used to reduce noise or create a smoothing effect.
3. **Saves the processed image** to a new file (e.g., `output.jpg`).

The Gaussian blur is a mathematical operation that averages pixel values in an image based on a weighted distribution (a Gaussian function). This results in a smooth, blurred effect.

---

### **Main Functionality**
1. **Image Loading**:
   - The code uses the `stb_image` library to load an image from a file. This library supports various image formats (e.g., JPEG, PNG) and handles the decoding process.
   - The image data is stored in a custom `Image` class, which represents the image as a collection of RGB pixel values.

2. **Gaussian Blur**:
   - The code is designed to apply a Gaussian blur to the image. Although the actual blur implementation is not shown in the provided code snippet, the structure is set up to handle it.
   - The blur is controlled by a parameter called `sigma`, which determines the strength of the blur. A higher `sigma` value results in a more pronounced blur effect.

3. **Image Saving**:
   - After processing, the blurred image is saved to a new file using the `stb_image_write` library. This library supports saving images in various formats.

4. **Command-Line Interface**:
   - The program is designed to be run from the command line. It takes input and output filenames as arguments, along with an optional `sigma` value for the Gaussian blur.

---

### **Algorithms Used**
1. **Gaussian Blur**:
   - The Gaussian blur is a convolution operation that applies a Gaussian kernel to the image. Each pixel's value is replaced by a weighted average of its neighboring pixels, where the weights are determined by the Gaussian function:
     \[
     G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
     \]
   - The `sigma` parameter controls the spread of the Gaussian function, which determines how much neighboring pixels influence the result.

2. **Image Loading and Saving**:
   - The `stb_image` and `stb_image_write` libraries handle the low-level details of reading and writing image files. These libraries are lightweight and widely used in C++ for image processing.

---

### **Overall Structure**
The code is organized into the following components:

1. **Image Class**:
   - Represents an image with properties like `width`, `height`, and `channels` (e.g., RGB or RGBA).
   - Contains methods for loading an image from a file and managing the image data.

2. **Main Function**:
   - Handles command-line arguments to specify the input file, output file, and `sigma` value.
   - Loads the input image, applies the Gaussian blur, and saves the result.

3. **Libraries**:
   - `stb_image.h`: For loading images.
   - `stb_image_write.h`: For saving images.
   - Standard C++ libraries (`<iostream>`, `<vector>`, `<cmath>`, `<string>`, `<algorithm>`) for general-purpose functionality.

---

### **How the Parts Work Together**
1. **Command-Line Input**:
   - The program starts by checking the command-line arguments. If the user provides insufficient arguments, it displays a usage message and exits.

2. **Image Loading**:
   - The `Image` class uses the `stb_image` library to load the image data into memory. The image is stored as a vector of `unsigned char` values, where each pixel's RGB values are stored sequentially.

3. **Gaussian Blur**:
   - The `sigma` value is parsed from the command line (or defaults to 2.0). This value is used to control the strength of the Gaussian blur.

4. **Image Saving**:
   - After applying the Gaussian blur, the processed image is saved to the specified output file using the `stb_image_write` library.

---

### **Problem Being Solved**
The code solves the problem of **image smoothing** or **noise reduction** using Gaussian blur. This is a common task in image processing, used in applications like:
- Enhancing image quality by reducing noise.
- Creating artistic effects.
- Preprocessing images for further analysis (e.g., edge detection).

---

### **Approach Taken**
1. **Modular Design**:
   - The `Image` class encapsulates all image-related functionality, making the code modular and reusable.

2. **Error Handling**:
   - The code includes error handling for invalid command-line arguments and image loading failures.

3. **Flexibility**:
   - The program allows the user to specify the strength of the Gaussian blur via the `sigma` parameter, making it adaptable to different use cases.

---

### **Summary**
This code is a **command-line image processing tool** that loads an image, applies a Gaussian blur, and saves the result. It uses the `stb_image` libraries for image I/O and a custom `Image` class to manage image data. The Gaussian blur is controlled by the `sigma` parameter, which determines the strength of the smoothing effect. The code is well-structured, modular, and designed for ease of use and flexibility.

Let me know if you'd like a line-by-line explanation or suggestions for improvements!