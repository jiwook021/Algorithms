# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in detail, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Header Files and Libraries**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
```
#### What It Does:
These lines include standard C++ libraries that provide essential functionality:
- `<iostream>`: For input/output operations (e.g., printing to the console).
- `<vector>`: For using the `std::vector` container, which is a dynamic array.
- `<cmath>`: For mathematical functions (e.g., `sqrt`, `exp`).
- `<string>`: For working with strings (e.g., storing filenames).
- `<algorithm>`: For common algorithms (e.g., sorting, searching).

#### Why It’s Used:
These libraries are included because the program needs to:
- Print messages to the console (`<iostream>`).
- Store image data dynamically (`<vector>`).
- Perform mathematical calculations for Gaussian blur (`<cmath>`).
- Handle filenames and command-line arguments (`<string>`).
- Use algorithms for processing data (`<algorithm>`).

---

### **2. STB Image Libraries**
```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
```
#### What It Does:
These lines include the `stb_image` and `stb_image_write` libraries, which are lightweight libraries for loading and saving images.

- `STB_IMAGE_IMPLEMENTATION` and `STB_IMAGE_WRITE_IMPLEMENTATION`: These macros tell the libraries to include their implementation code.
- `stb_image.h`: For loading images from files.
- `stb_image_write.h`: For saving images to files.

#### Why It’s Used:
These libraries are used because:
- They are simple and lightweight, making them ideal for basic image processing tasks.
- They support multiple image formats (e.g., JPEG, PNG) without requiring complex dependencies.

---

### **3. Image Class**
```cpp
class Image {
private:
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data; // Stores RGB values sequentially
```
#### What It Does:
This defines a class called `Image` that represents an image. It has:
- **Private Members**:
  - `width`: The width of the image in pixels.
  - `height`: The height of the image in pixels.
  - `channels`: The number of color channels (e.g., 3 for RGB, 4 for RGBA).
  - `data`: A vector that stores the pixel data. Each pixel’s RGB values are stored sequentially.

#### Why It’s Used:
The `Image` class encapsulates all the properties and functionality related to an image. This makes the code modular and easier to maintain.

---

### **4. Constructors**
```cpp
public:
    // Default constructor
    Image() : width(0), height(0), channels(0) {}
    
    // Constructor for creating an image with specific dimensions
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(width * height * channels, 0);
    }
    
    // Copy constructor
    Image(const Image& other) : width(other.width), height(other.height), 
                               channels(other.channels), data(other.data) {}
```
#### What It Does:
- **Default Constructor**: Initializes an empty image with `width`, `height`, and `channels` set to 0.
- **Parameterized Constructor**: Creates an image with specific dimensions and allocates memory for the pixel data.
- **Copy Constructor**: Creates a copy of an existing image.

#### Why It’s Used:
- **Default Constructor**: Used when an image object is created without specifying dimensions.
- **Parameterized Constructor**: Used to create an image with specific dimensions.
- **Copy Constructor**: Ensures that copying an image object creates a deep copy of its data.

---

### **5. Image Loading**
```cpp
bool load(const std::string& filename) {
    // Free any existing image data
    if (!data.empty()) {
        data.clear();
    }
    
    // Load image using stb_image
    int w, h, c;
    unsigned char* imgData = stbi_load(filename.c_str(), &w, &h, &c, 0);
    
    if (!imgData) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    
    // Set image properties
    width = w;
    height = h;
    channels = c;
    
    // Copy image data to our vector
    data.assign(imgData, imgData + width * height * channels);
    
    // Free the original image data
    stbi_image_free(imgData);
    
    return true;
}
```
#### What It Does:
This function loads an image from a file:
1. Clears any existing image data.
2. Uses `stbi_load` to load the image file.
3. Checks if the image was loaded successfully.
4. Sets the image properties (`width`, `height`, `channels`).
5. Copies the image data into the `data` vector.
6. Frees the memory allocated by `stbi_load`.

#### Why It’s Used:
- **Error Handling**: If the image fails to load, an error message is printed, and the function returns `false`.
- **Memory Management**: The `data` vector is used to store the image data, ensuring it is managed safely.

---

### **6. Main Function**
```cpp
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.jpg [sigma]" << std::endl;
        std::cerr << "  sigma: Standard deviation for Gaussian blur (default: 2.0)" << std::endl;
        return 1;
    }
```
#### What It Does:
- Checks if the user provided at least two command-line arguments (`input.jpg` and `output.jpg`).
- If not, it prints a usage message and exits.

#### Why It’s Used:
- Ensures the program is used correctly by requiring the necessary arguments.

---

### **7. Parsing Command-Line Arguments**
```cpp
    // Get input and output filenames
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    // Get sigma value (optional)
    double sigma = 2.0; // Default sigma value
    if (argc > 3) {
        try {
            sigma = std::stod(argv[3]);
            if (sigma <= 0) {
                throw std::invalid_argument("Sigma must be positive");
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid sigma value: " << e.what() << std::endl;
            return 1;
        }
    }
```
#### What It Does:
- Extracts the input and output filenames from the command-line arguments.
- Parses the optional `sigma` value, defaulting to 2.0 if not provided.
- Validates that `sigma` is a positive number.

#### Why It’s Used:
- Provides flexibility by allowing the user to specify the strength of the Gaussian blur.

---

### **8. Loading the Input Image**
```cpp
    // Load input image
    Image inputImage;
    if (!inputImage.load(inputFile)) {
        return 1;
    }
```
#### What It Does:
- Creates an `Image` object and loads the input image from the specified file.
- If loading fails, the program exits.

#### Why It’s Used:
- Prepares the image for processing by loading it into memory.

---

### **9. Applying Gaussian Blur**
```cpp
    // Apply Gaussian blur
```
#### What It Does:
- This section is incomplete in the provided code, but it would apply a Gaussian blur to the image using the specified `sigma` value.

#### Why It’s Used:
- Gaussian blur is a common technique for smoothing images and reducing noise.

---

### **Summary**
This code is a **command-line image processing tool** that loads an image, applies a Gaussian blur, and saves the result. It uses the `stb_image` libraries for image I/O and a custom `Image` class to manage image data. The Gaussian blur is controlled by the `sigma` parameter, which determines the strength of the smoothing effect. The code is well-structured, modular, and designed for ease of use and flexibility.

Let me know if you’d like further clarification or a deeper dive into any specific part!