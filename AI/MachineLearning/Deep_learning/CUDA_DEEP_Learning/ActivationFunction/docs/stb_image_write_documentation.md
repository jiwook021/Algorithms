# stb_image_write.h Documentation

## Overview

The `stb_image_write.h` header file is a public domain library designed for writing images in various formats (PNG, BMP, TGA, JPEG, HDR) to C standard I/O or through a custom callback function. Authored by Sean Barrett and contributors, this library prioritizes simplicity and compactness of source code over optimal image file size or runtime performance. It is particularly useful for applications that require straightforward image output capabilities without the need for extensive image processing features.

## Key Functions

### Image Writing Functions

The library provides functions to write images in different formats. Each function returns `0` on failure and a non-zero value on success.

- **PNG Writing**
  ```c
  int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
  ```
  Writes a PNG file. Supports non-consecutive row data through `stride_in_bytes`.

- **BMP Writing**
  ```c
  int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
  ```
  Writes a BMP file. Converts monochrome (Y) to RGB and does not support alpha channels.

- **TGA Writing**
  ```c
  int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
  ```
  Writes a TGA file. Supports RLE compression, which can be disabled.

- **JPEG Writing**
  ```c
  int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
  ```
  Writes a JPEG file. Ignores alpha channels and allows quality settings from 1 to 100.

- **HDR Writing**
  ```c
  int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
  ```
  Writes an HDR file. Expects linear float data and discards alpha channels.

### Custom Callback Functions

For each image format, there are equivalent functions that use a custom write function:

```c
typedef void stbi_write_func(void *context, void *data, int size);
```

- **Example for PNG:**
  ```c
  int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes);
  ```

These functions allow writing to non-standard output streams by providing a custom function to handle the data.

### Utility Functions

- **Vertical Flip**
  ```c
  void stbi_flip_vertically_on_write(int flag);
  ```
  Flips the image data vertically if `flag` is non-zero.

## Algorithm Analysis

The library focuses on simplicity and compactness rather than performance. The PNG compression is not optimal, resulting in larger file sizes compared to other implementations. The complexity of the functions is generally linear with respect to the number of pixels, as each pixel is processed once.

### Complexity

- **Time Complexity:** O(n), where n is the number of pixels.
- **Space Complexity:** O(1) additional space, aside from the input data.

## Dependencies and Interactions

- **Standard Library:** Uses standard C library functions for file I/O unless `STBI_WRITE_NO_STDIO` is defined.
- **Memory Management:** Custom memory management functions can be defined to replace `malloc`, `realloc`, and `free`.

## Usage Examples

### Writing a PNG File

```c
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int width = 512, height = 512;
    unsigned char *image_data = generate_image_data(width, height); // Assume this function generates image data
    stbi_write_png("output.png", width, height, 3, image_data, width * 3);
    free(image_data);
    return 0;
}
```

### Using a Custom Write Function

```c
void custom_write_func(void *context, void *data, int size) {
    FILE *file = (FILE *)context;
    fwrite(data, 1, size, file);
}

int main() {
    FILE *file = fopen("output.png", "wb");
    int width = 512, height = 512;
    unsigned char *image_data = generate_image_data(width, height);
    stbi_write_png_to_func(custom_write_func, file, width, height, 3, image_data, width * 3);
    fclose(file);
    free(image_data);
    return 0;
}
```

## Potential Issues and Limitations

- **Strict Aliasing:** The library may not work correctly with strict aliasing optimizations enabled.
- **PNG Compression:** The built-in PNG compression is not optimal. Users can provide a custom zlib compression function to improve this.
- **HDR Writing:** Requires standard I/O for formatted output, which is disabled if `STBI_WRITE_NO_STDIO` is defined.
- **JPEG Alpha Channels:** JPEG writing ignores alpha channels, which may not be suitable for all use cases.
- **Memory Management:** Users must ensure that custom memory management functions are correctly implemented if the defaults are overridden.

## Conclusion

`stb_image_write.h` is a versatile and easy-to-use library for writing images in multiple formats. While it may not offer the most efficient compression, its simplicity and ease of integration make it a valuable tool for developers needing basic image output capabilities.