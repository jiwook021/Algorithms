# stb_image_write.h Documentation

## Overview

The `stb_image_write.h` header file is a public domain library designed for writing images in various formats, including PNG, BMP, TGA, JPEG, and HDR. It provides a simple API for exporting image data to files or through custom write functions. The library prioritizes simplicity and ease of integration over optimal file size and performance, making it suitable for applications where these factors are not critical.

## Key Functions

### Image Writing Functions

Each of the following functions writes an image to a file in the specified format:

- **`int stbi_write_png(const char *filename, int w, int h, int comp, const void *data, int stride_in_bytes);`**
  - Writes a PNG image to a file.
  - **Parameters:**
    - `filename`: Path to the output file.
    - `w`, `h`: Width and height of the image.
    - `comp`: Number of color components per pixel (1=Y, 2=YA, 3=RGB, 4=RGBA).
    - `data`: Pointer to the image data.
    - `stride_in_bytes`: Byte distance between the start of consecutive rows.
  - **Returns:** Non-zero on success, 0 on failure.

- **`int stbi_write_bmp(const char *filename, int w, int h, int comp, const void *data);`**
  - Writes a BMP image to a file.
  - **Parameters:** Similar to `stbi_write_png`, but without `stride_in_bytes`.

- **`int stbi_write_tga(const char *filename, int w, int h, int comp, const void *data);`**
  - Writes a TGA image to a file.
  - **Parameters:** Similar to `stbi_write_bmp`.

- **`int stbi_write_jpg(const char *filename, int w, int h, int comp, const void *data, int quality);`**
  - Writes a JPEG image to a file.
  - **Parameters:**
    - Additional `quality` parameter (1-100) to control compression quality.

- **`int stbi_write_hdr(const char *filename, int w, int h, int comp, const float *data);`**
  - Writes an HDR image to a file.
  - **Parameters:** Similar to `stbi_write_bmp`, but `data` is a `float` pointer.

### Custom Write Functions

These functions allow writing images using a custom callback function:

- **`int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes);`**
  - Writes a PNG image using a custom function.
  - **Parameters:** Similar to `stbi_write_png`, with `func` and `context` for custom writing.

- **Other formats:** Similar functions exist for BMP, TGA, JPEG, and HDR.

### Utility Functions

- **`void stbi_flip_vertically_on_write(int flag);`**
  - Flips the image vertically before writing.
  - **Parameters:** `flag` is non-zero to enable flipping.

## Algorithm Analysis

- **Complexity:** The complexity of image writing functions is generally linear with respect to the number of pixels (`O(w * h)`), as each pixel is processed once.
- **Approach:** The library uses straightforward algorithms to convert image data into the desired file format. PNG compression can be customized for better performance using a custom zlib function.

## Dependencies and Interactions

- **Standard Library:** The library uses standard C functions for file I/O and memory management. Custom memory functions can be defined to replace `malloc`, `realloc`, and `free`.
- **Custom Compression:** For PNG, a custom zlib-style compression function can be provided to optimize file size.

## Usage Examples

```c
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int width = 512, height = 512, channels = 3;
    unsigned char *image_data = ...; // Assume this is initialized with image data

    // Write a PNG file
    if (!stbi_write_png("output.png", width, height, channels, image_data, width * channels)) {
        fprintf(stderr, "Failed to write PNG\n");
    }

    // Write a JPEG file with quality 90
    if (!stbi_write_jpg("output.jpg", width, height, channels, image_data, 90)) {
        fprintf(stderr, "Failed to write JPEG\n");
    }

    return 0;
}
```

## Potential Issues, Edge Cases, and Limitations

- **File Size:** PNG files written by this library may be larger than those produced by more optimized libraries.
- **Strict Aliasing:** The library may not work correctly with strict-aliasing optimizations enabled.
- **Alpha Channel Handling:** BMP and JPEG formats do not support alpha channels, and HDR discards alpha data.
- **Stride Handling:** Only PNG format supports non-consecutive row data via stride; other formats require contiguous data.
- **JPEG Quality:** The quality parameter for JPEG affects file size and image fidelity, with higher values producing larger files.
- **Platform-Specific Issues:** On Windows, UTF-8 encoded filenames require defining `STBIW_WINDOWS_UTF8`.

## Conclusion

The `stb_image_write.h` library provides a simple and flexible interface for writing images in multiple formats. While it may not produce the smallest files or offer the best performance, its ease of use and minimal dependencies make it an excellent choice for many applications.