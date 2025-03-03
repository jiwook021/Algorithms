// Project UID af1f95f547e44c8ea88730dfb185559d

#include <cassert>
#include "Image.h"
#include "Matrix.h"

// REQUIRES: img points to an Image
//           0 < width && width <= MAX_MATRIX_WIDTH
//           0 < height && height <= MAX_MATRIX_HEIGHT
// MODIFIES: *img
// EFFECTS:  Initializes the Image with the given width and height.
// NOTE:     Do NOT use new or delete here.
void Image_init(Image* Img, int width, int Height) {
  assert(0 < width && width <= MAX_MATRIX_WIDTH);
  assert(0 < Height && Height <= MAX_MATRIX_HEIGHT);
  Img->width = width;
  Img->Height = Height;
  Matrix_init(&Img->RedChannel, width, Height);
  Matrix_init(&Img->GreenChannel, width, Height);
  Matrix_init(&Img->BlueChannel, width, Height);
}

// REQUIRES: img points to an Image
//           is contains an image in PPM format without comments
//           (any kind of whitespace is ok)
// MODIFIES: *img
// EFFECTS:  Initializes the Image by reading in an image in PPM format
//           from the given input stream.
// NOTE:     See the project spec for a discussion of PPM format.
// NOTE:     Do NOT use new or delete here.
void Image_init(Image* Img, std::istream& Is) {
  std::string Trash = "";
  Is >> Trash;
  Is >> Img->width;
  Is >> Img->Height;
  Is >> Trash;
  Matrix_init(&Img->RedChannel, Img->width, Img->Height);
  Matrix_init(&Img->GreenChannel, Img->width, Img->Height);
  Matrix_init(&Img->BlueChannel, Img->width, Img->Height);

  for(int i = 0; i < Img->Height; i++) {
    for(int j = 0; j < Img->width; j++) {
      Is >> * Matrix_at(&Img->RedChannel, i, j); 
      Is >> * Matrix_at(&Img->GreenChannel, i, j); 
      Is >> * Matrix_at(&Img->BlueChannel, i, j);
    }
  }
}

// REQUIRES: img points to a valid Image
// EFFECTS:  Writes the image to the given output stream in PPM format.
//           You must use the kind of whitespace specified here.
//           First, prints out the header for the image like this:
//             P3 [newline]
//             WIDTH [space] HEIGHT [newline]
//             255 [newline]
//           Next, prints out the rows of the image, each followed by a
//           newline. Each pixel in a row is printed as three ints
//           for its red, green, and blue components, in that order. Each
//           int is followed by a space. This means that there will be an
//           "extra" space at the end of each line. See the project spec
//           for an example.
void Image_print(const Image* Img, std::ostream& Os) {
   Os << "P3" << "\n";
   Os << Img->width << " " << Img->Height << "\n";
   Os << MAX_INTENSITY << "\n";

   for(int i = 0; i < Img->Height; i++) {
     for(int j = 0; j < Img->width; j++) {
        Os << * Matrix_at(&Img->RedChannel, i, j) << " ";
        Os << * Matrix_at(&Img->GreenChannel, i, j) << " ";
        Os << * Matrix_at(&Img->BlueChannel, i, j) << " ";
     }
     Os << "\n";
   }
}

// REQUIRES: img points to a valid Image
// EFFECTS:  Returns the width of the Image.
int Image_width(const Image* Img) {
  return Img->width;
}

// REQUIRES: img points to a valid Image
// EFFECTS:  Returns the height of the Image.
int Image_height(const Image* Img) {
  return Img->Height;
}

// REQUIRES: img points to a valid Image
//           0 <= row && row < Image_height(img)
//           0 <= column && column < Image_width(img)
// EFFECTS:  Returns the pixel in the Image at the given row and column.
Pixel Image_get_pixel(const Image* Img, int Row, int Column) {
  assert(0 <= Row && Row < Image_height(Img));
  assert(0 <= Column && Column < Image_width(Img));

  int r = * Matrix_at(&Img->RedChannel, Row, Column);
  int g = * Matrix_at(&Img->GreenChannel, Row, Column);
  int b = * Matrix_at(&Img->BlueChannel, Row, Column);
  Pixel Pix = {r, g, b};
  return Pix;
}

// REQUIRES: img points to a valid Image
//           0 <= row && row < Image_height(img)
//           0 <= column && column < Image_width(img)
// MODIFIES: *img
// EFFECTS:  Sets the pixel in the Image at the given row and column
//           to the given color.
void Image_set_pixel(Image* Img, int Row, int Column, Pixel Color) {
  assert(0 <= Row && Row < Image_height(Img));
  assert(0 <= Column && Column < Image_width(Img));

  * Matrix_at(&Img->RedChannel, Row, Column) = Color.r;
  * Matrix_at(&Img->GreenChannel, Row, Column) = Color.g;
  * Matrix_at(&Img->BlueChannel, Row, Column) = Color.b;
}

// REQUIRES: img points to a valid Image
// MODIFIES: *img
// EFFECTS:  Sets each pixel in the image to the given color.
void Image_fill(Image* Img, Pixel Color) {
  Matrix_fill(&Img->RedChannel, Color.r);
  Matrix_fill(&Img->GreenChannel, Color.g);
  Matrix_fill(&Img->BlueChannel, Color.b);
}

