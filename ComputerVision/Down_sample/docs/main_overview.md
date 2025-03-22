# Code Overview: main.cpp

This C++ code is an **image processing program** that performs **image downsampling** (reducing the resolution of an image) using a multi-threaded approach. Let’s break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The program takes an input image, reduces its resolution by half (downsampling), and saves the resulting image. The downsampling process averages the color values of 2x2 pixel blocks from the original image to create a single pixel in the downsampled image. This is a common technique used in image processing to reduce image size while preserving visual quality.

The program also leverages **multi-threading** to speed up the downsampling process by dividing the work across multiple CPU cores.

---

### **Main Functionality**
1. **Image Loading**:
   - The program loads an image file (e.g., `input.png`) using the `stb_image` library.
   - The image is stored in memory as a vector of RGBA (Red, Green, Blue, Alpha) pixel values.

2. **Downsampling**:
   - The image is downsampled by reducing its width and height by half.
   - For each pixel in the downsampled image, the program computes the average color value of a 2x2 block of pixels from the original image.

3. **Multi-threading**:
   - The downsampling process is parallelized using multiple threads. The work is divided such that each thread processes a portion of the image.

4. **Image Saving**:
   - The downsampled image is saved to a file (e.g., `output.png`) using the `stb_image_write` library.

---

### **Algorithms Used**
1. **Downsampling Algorithm**:
   - The program uses a **block averaging** algorithm. For each pixel in the downsampled image:
     - It identifies the corresponding 2x2 block of pixels in the original image.
     - It calculates the average of the RGBA values for these 4 pixels.
     - The resulting average is assigned to the corresponding pixel in the downsampled image.

2. **Multi-threading Algorithm**:
   - The program divides the image into horizontal strips, with each strip assigned to a separate thread.
   - The number of threads is determined by the number of CPU cores available on the system.
   - Each thread processes its assigned portion of the image independently.

---

### **Overall Structure**
The code is organized into the following components:

1. **Dependencies**:
   - The program uses the `stb_image` and `stb_image_write` libraries for image loading and saving.
   - It also uses the C++ Standard Library for multi-threading (`<thread>`) and memory management (`<memory>`).

2. **Image Class**:
   - The `Image` class encapsulates the image data and provides methods for loading, saving, and downsampling images.
   - Key data members:
     - `width_`, `height_`: Dimensions of the image.
     - `channels_`: Number of color channels (always 4 for RGBA).
     - `pixels_`: A vector storing the pixel data in RGBA format.
   - Key methods:
     - `Image(const std::string& filename)`: Constructor to load an image from a file.
     - `Image(int w, int h)`: Constructor to create a blank image of a given size.
     - `Save(const std::string& filename)`: Saves the image to a file.
     - `DownSample()`: Downsamples the image using multi-threading.
     - `ComputeAvgPixel(int dstX, int dstY, const Image& src)`: Computes the average pixel value for a 2x2 block.

3. **Main Function**:
   - The `main()` function orchestrates the program:
     - Loads the input image.
     - Downsamples the image.
     - Saves the downsampled image.
     - Handles errors gracefully.

---

### **How the Parts Work Together**
1. **Image Loading**:
   - The `Image` constructor loads the image file into memory using `stbi_load()`.
   - The pixel data is stored in the `pixels_` vector.

2. **Downsampling**:
   - The `DownSample()` method creates a new `Image` object with half the width and height of the original.
   - It divides the work among multiple threads, with each thread processing a portion of the image.
   - The `ComputeAvgPixel()` method calculates the average pixel value for each 2x2 block.

3. **Multi-threading**:
   - The program determines the number of available CPU cores using `std::thread::hardware_concurrency()`.
   - It divides the image into horizontal strips and assigns each strip to a thread.
   - The threads work in parallel to compute the downsampled image.

4. **Image Saving**:
   - The `Save()` method writes the downsampled image to a file using `stbi_write_png()`.

5. **Error Handling**:
   - The program uses exception handling to catch and report errors (e.g., failed image loading or saving).

---

### **Problem Being Solved**
The program solves the problem of **efficiently reducing the resolution of an image** while maintaining visual quality. By using multi-threading, it ensures that the downsampling process is fast and scalable, especially for large images.

---

### **Approach Taken**
1. **Block Averaging**:
   - The downsampling process averages 2x2 blocks of pixels, which is a simple and effective way to reduce image size while preserving detail.

2. **Multi-threading**:
   - The program leverages modern multi-core CPUs to speed up the computation by dividing the work among multiple threads.

3. **Memory Management**:
   - The program uses `std::unique_ptr` to manage dynamically allocated memory, ensuring that resources are properly cleaned up.

4. **Error Handling**:
   - The program uses exceptions to handle errors gracefully, providing meaningful error messages to the user.

---

### **Summary**
This code is a well-structured, multi-threaded image downsampling program. It uses block averaging to reduce image resolution and leverages multi-threading to improve performance. The program is designed to be efficient, scalable, and easy to use, making it suitable for processing large images quickly.

Let me know if you'd like to dive into the line-by-line explanation or discuss potential improvements!