# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also explain the **why** behind each design choice.

---

### **1. Preprocessor Directives and Includes**
```cpp
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include "stb_image.h"
#include "stb_image_write.h"
```

#### **What It Does**
- These lines prepare the program to use external libraries and include necessary C++ standard libraries.

#### **Explanation**
1. **Preprocessor Directives**:
   - `#define STB_IMAGE_IMPLEMENTATION` and `#define STB_IMAGE_WRITE_IMPLEMENTATION`:
     - These tell the `stb_image` and `stb_image_write` libraries to include their implementation code in this file.
     - Without these, the libraries wouldn’t work.

2. **Includes**:
   - `<iostream>`: Provides input/output functionality (e.g., `std::cout` for printing to the console).
   - `<vector>`: Provides the `std::vector` class, which is a dynamic array (a resizable list of elements).
   - `<thread>`: Provides multi-threading support (e.g., creating and managing threads).
   - `<memory>`: Provides smart pointers like `std::unique_ptr`, which automatically manage memory.
   - `"stb_image.h"` and `"stb_image_write.h"`: External libraries for loading and saving images.

#### **Why These Are Used**
- The `stb_image` and `stb_image_write` libraries are lightweight and easy to use for image processing.
- The C++ standard libraries provide essential tools for memory management, multi-threading, and I/O.

---

### **2. Image Class Definition**
```cpp
class Image {
public:
    Image(const std::string& filename);
    Image(int w, int h);

    bool Save(const std::string& filename) const;
    std::unique_ptr<Image> DownSample() const;

private:
    int width_;
    int height_;
    int channels_;
    std::vector<uint8_t> pixels_; // RGBA pixels

    void ComputeAvgPixel(int dstX, int dstY, const Image& src);
};
```

#### **What It Does**
- Defines a class called `Image` that represents an image and provides methods to load, save, and downsample it.

#### **Explanation**
1. **Public Methods**:
   - `Image(const std::string& filename)`: Constructor to load an image from a file.
   - `Image(int w, int h)`: Constructor to create a blank image of a given size.
   - `Save(const std::string& filename)`: Saves the image to a file.
   - `DownSample()`: Downsamples the image (reduces its resolution).

2. **Private Data Members**:
   - `width_`, `height_`: Dimensions of the image.
   - `channels_`: Number of color channels (always 4 for RGBA).
   - `pixels_`: A vector storing the pixel data in RGBA format.

3. **Private Method**:
   - `ComputeAvgPixel(int dstX, int dstY, const Image& src)`: Computes the average pixel value for a 2x2 block.

#### **Why This Structure Is Used**
- The `Image` class encapsulates all image-related data and functionality, making the code modular and reusable.
- Private members ensure that the internal representation of the image is hidden from the outside world, which is a good practice in object-oriented programming.

---

### **3. Image Constructors**
#### **Constructor 1: Loading an Image**
```cpp
Image::Image(const std::string& filename) {
    unsigned char* data = stbi_load(filename.c_str(), &width_, &height_, &channels_, 4);
    if (!data) throw std::runtime_error("Failed to load image: " + filename);
    channels_ = 4;
    pixels_.assign(data, data + width_ * height_ * channels_);
    stbi_image_free(data);
}
```

#### **What It Does**
- Loads an image from a file and stores its pixel data in the `pixels_` vector.

#### **Explanation**
1. **Loading the Image**:
   - `stbi_load(filename.c_str(), &width_, &height_, &channels_, 4)`:
     - Loads the image from the file specified by `filename`.
     - `width_`, `height_`, and `channels_` are filled with the image’s dimensions and number of color channels.
     - The `4` forces the image to be loaded as RGBA (4 channels).

2. **Error Handling**:
   - If `stbi_load` fails (returns `nullptr`), the program throws an exception with an error message.

3. **Storing Pixel Data**:
   - `pixels_.assign(data, data + width_ * height_ * channels_)`:
     - Copies the pixel data from `data` (a raw array) into the `pixels_` vector.

4. **Freeing Memory**:
   - `stbi_image_free(data)`:
     - Frees the memory allocated by `stbi_load`.

#### **Why This Approach Is Used**
- `stbi_load` is a simple and efficient way to load images.
- Using a `std::vector` for `pixels_` ensures that memory is managed automatically.

---

#### **Constructor 2: Creating a Blank Image**
```cpp
Image::Image(int w, int h) : width_(w), height_(h), channels_(4), pixels_(w * h * 4) {}
```

#### **What It Does**
- Creates a blank image of a specified size.

#### **Explanation**
1. **Initialization List**:
   - `width_(w)`, `height_(h)`, `channels_(4)`: Initializes the image dimensions and channels.
   - `pixels_(w * h * 4)`: Allocates space for the pixel data (4 bytes per pixel for RGBA).

#### **Why This Approach Is Used**
- This constructor is used to create a blank image for storing the downsampled result.

---

### **4. Save Method**
```cpp
bool Image::Save(const std::string& filename) const {
    return stbi_write_png(filename.c_str(), width_, height_, channels_, pixels_.data(), width_ * channels_);
}
```

#### **What It Does**
- Saves the image to a file in PNG format.

#### **Explanation**
1. **Saving the Image**:
   - `stbi_write_png(filename.c_str(), width_, height_, channels_, pixels_.data(), width_ * channels_)`:
     - Writes the image data to a file.
     - `pixels_.data()` provides a pointer to the pixel data.
     - `width_ * channels_` specifies the number of bytes per row.

#### **Why This Approach Is Used**
- `stbi_write_png` is a simple and efficient way to save images in PNG format.

---

### **5. ComputeAvgPixel Method**
```cpp
void Image::ComputeAvgPixel(int dstX, int dstY, const Image& src) {
    int srcX = dstX * 2;
    int srcY = dstY * 2;
    int sum[4] = {0, 0, 0, 0};
    int count = 0;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int ix = srcX + dx;
            int iy = srcY + dy;
            if (ix < src.width_ && iy < src.height_) {
                const uint8_t* p = &src.pixels_[(iy * src.width_ + ix) * 4];
                for (int c = 0; c < 4; c++)
                    sum[c] += p[c];
                count++;
            }
        }
    }

    uint8_t* dstPixel = &pixels_[(dstY * width_ + dstX) * 4];
    for (int c = 0; c < 4; c++)
        dstPixel[c] = static_cast<uint8_t>(sum[c] / count);
}
```

#### **What It Does**
- Computes the average color value of a 2x2 block of pixels from the source image and assigns it to a pixel in the downsampled image.

#### **Explanation**
1. **Source Pixel Coordinates**:
   - `srcX = dstX * 2` and `srcY = dstY * 2`:
     - Maps the destination pixel coordinates to the corresponding 2x2 block in the source image.

2. **Summing Pixel Values**:
   - `sum[4]` stores the sum of the RGBA values for the 2x2 block.
   - `count` keeps track of the number of valid pixels in the block.

3. **Nested Loops**:
   - The outer loop (`dy`) and inner loop (`dx`) iterate over the 2x2 block.
   - `ix` and `iy` are the coordinates of the current pixel in the source image.

4. **Boundary Check**:
   - `if (ix < src.width_ && iy < src.height_)`:
     - Ensures that the coordinates are within the bounds of the source image.

5. **Pixel Access**:
   - `const uint8_t* p = &src.pixels_[(iy * src.width_ + ix) * 4]`:
     - Accesses the RGBA values of the current pixel.

6. **Averaging**:
   - The RGBA values are summed up and divided by `count` to compute the average.

7. **Assigning to Destination**:
   - The averaged values are assigned to the corresponding pixel in the downsampled image.

#### **Why This Approach Is Used**
- This method ensures that the downsampled image is a smooth representation of the original image.

---

### **6. DownSample Method**
```cpp
std::unique_ptr<Image> Image::DownSample() const {
    int newWidth = width_ / 2;
    int newHeight = height_ / 2;
    auto result = std::make_unique<Image>(newWidth, newHeight);

    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::vector<std::thread> threads;
    int rowsPerThread = newHeight / numThreads;
    int extra = newHeight % numThreads;

    auto worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
            for (int x = 0; x < newWidth; ++x)
                result->ComputeAvgPixel(x, y, *this);
    };

    int currentY = 0;
    for (unsigned i = 0; i < numThreads; ++i) {
        int startY = currentY;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(worker, startY, endY);
        currentY = endY;
    }

    for (auto& t : threads) t.join();
    return result;
}
```

#### **What It Does**
- Downsamples the image using multi-threading.

#### **Explanation**
1. **New Dimensions**:
   - `newWidth = width_ / 2` and `newHeight = height_ / 2`:
     - The downsampled image will have half the width and height of the original.

2. **Creating the Result Image**:
   - `auto result = std::make_unique<Image>(newWidth, newHeight)`:
     - Creates a blank image for the downsampled result.

3. **Multi-threading Setup**:
   - `numThreads = std::thread::hardware_concurrency()`:
     - Determines the number of CPU cores available.
   - `rowsPerThread = newHeight / numThreads` and `extra = newHeight % numThreads`:
     - Divides the work among threads.

4. **Worker Function**:
   - The `worker` lambda function processes a range of rows in the downsampled image.

5. **Creating Threads**:
   - Each thread is assigned a range of rows to process.

6. **Joining Threads**:
   - `t.join()` ensures that all threads complete their work before the function returns.

#### **Why This Approach Is Used**
- Multi-threading speeds up the downsampling process by utilizing multiple CPU cores.

---

### **7. Main Function**
```cpp
int main() {
    try {
        Image img("input.png");
        auto downsampled = img.DownSample();
        if (downsampled->Save("output.png"))
            std::cout << "Saved successfully!\n";
        else
            std::cerr << "Failed to save image!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return -1;
    }
    return 0;
}
```

#### **What It Does**
- Loads an image, downsamples it, and saves the result.

#### **Explanation**
1. **Loading the Image**:
   - `Image img("input.png")`:
     - Loads the image from `input.png`.

2. **Downsampling**:
   - `auto downsampled = img.DownSample()`:
     - Downsamples the image.

3. **Saving the Image**:
   - `downsampled->Save("output.png")`:
     - Saves the downsampled image to `output.png`.

4. **Error Handling**:
   - If an error occurs, it is caught and reported.

#### **Why This Approach Is Used**
- The `main()` function orchestrates the program and handles errors gracefully.

---

### **Summary**
This code is a well-structured, multi-threaded image downsampling program. It uses block averaging to reduce image resolution and leverages multi-threading to improve performance. The program is designed to be efficient, scalable, and easy to use, making it suitable for processing large images quickly.

Let me know if you’d like to discuss potential improvements!