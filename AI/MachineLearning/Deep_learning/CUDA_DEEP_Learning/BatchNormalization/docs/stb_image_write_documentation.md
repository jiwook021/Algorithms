# stb_image_write.h Documentation

## Overview

The `stb_image_write.h` header file is a public domain library designed for writing images in various formats (PNG, BMP, TGA, JPEG, HDR) to C stdio or a user-defined callback function. Created by Sean Barrett, this library prioritizes simplicity and compactness of source code over optimal image file size or runtime performance. It provides a straightforward API for developers to output images from raw pixel data, making it a useful tool for applications that need to export images without relying on complex external libraries.

## Key Functions and Classes

### Image Writing Functions

The library provides a set of functions to write images in different formats:

- **PNG Writing**:
  ```c
  int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
  ```
  Writes a PNG image to a file. The `stride_in_bytes` parameter allows writing sub-rectangles of a larger image.

- **BMP Writing**:
  ```c
  int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
  ```
  Writes a BMP image to a file. Note that BMP expands monochrome data to RGB and does not support alpha channels.

- **TGA Writing**:
  ```c
  int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
  ```
  Writes a TGA image to a file. Supports both RLE and non-RLE compression.

- **JPEG Writing**:
  ```c
  int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
  ```
  Writes a JPEG image to a file. The `quality` parameter ranges from 1 to 100, affecting the compression level and image quality.

- **HDR Writing**:
  ```c
  int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
  ```
  Writes an HDR image to a file. Expects linear float data and discards alpha channels.

### Callback-Based Writing Functions

These functions allow writing images using a custom callback function instead of directly to a file:

- **PNG to Function**:
  ```c
  int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes);
  ```

- **BMP to Function**:
  ```c
  int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
  ```

- **TGA to Function**:
  ```c
  int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
  ```

- **JPEG to Function**:
  ```c
  int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data, int quality);
  ```

- **HDR to Function**:
  ```c
  int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data);
  ```

### Utility Functions

- **Vertical Flip**:
  ```c
  void stbi_flip_vertically_on_write(int flag);
  ```
  Flips the image data vertically if the `flag` is non-zero.

### Global Configuration Variables

- **TGA RLE Compression**:
  ```c
  int stbi_write_tga_with_rle;
  ```
  Controls RLE compression for TGA files. Defaults to true.

- **PNG Compression Level**:
  ```c
  int stbi_write_png_compression_level;
  ```
  Sets the compression level for PNG files. Defaults to 8.

- **PNG Filter Mode**:
  ```c
  int stbi_write_force_png_filter;
  ```
  Forces a specific PNG filter mode. Defaults to -1 (automatic).

## Algorithm Analysis

The library uses straightforward algorithms for image writing, focusing on simplicity:

- **Complexity**: The complexity of writing an image is generally O(n), where n is the number of pixels, as each pixel is processed once.
- **Approach**: The library processes image data row by row, writing it to the output format. For PNG, it supports non-contiguous row data using the `stride_in_bytes` parameter.

## Dependencies and Interactions

- **Standard Library**: The library uses standard C functions for file operations and memory management. It can be configured to use custom memory allocation functions.
- **Custom Compression**: For PNG, a custom zlib-style compression function can be provided.

## Usage Examples

### Writing a PNG Image

```c
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int width = 800, height = 600, channels = 3;
    unsigned char *image_data = generate_image_data(width, height, channels);
    stbi_write_png("output.png", width, height, channels, image_data, width * channels);
    free(image_data);
    return 0;
}
```

### Writing a JPEG Image with Custom Quality

```c
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int width = 800, height = 600, channels = 3;
    unsigned char *image_data = generate_image_data(width, height, channels);
    stbi_write_jpg("output.jpg", width, height, channels, image_data, 90);
    free(image_data);
    return 0;
}
```

## Potential Issues, Edge Cases, and Limitations

- **Strict Aliasing**: The library may not work correctly with strict-aliasing optimizations enabled.
- **PNG Compression**: The default PNG compression is not optimal and may result in larger files compared to other libraries.
- **Alpha Channel Handling**: BMP and JPEG formats do not support alpha channels, and HDR discards them.
- **Memory Management**: Custom memory management functions can be defined, but incorrect implementations may lead to memory leaks or corruption.
- **Platform-Specific Issues**: On Windows, UTF-8 filenames require defining `STBIW_WINDOWS_UTF8` and converting filenames using `stbiw_convert_wchar_to_utf8`.

This documentation provides a comprehensive overview of the `stb_image_write.h` library, helping developers understand its functionality, usage, and potential pitfalls.