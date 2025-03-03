#include <cassert>
#include "processing.h"
#include "Image.h"
#include "Matrix.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cassert>


using namespace std;

// v DO NOT CHANGE v ------------------------------------------------
// The implementation of rotate_left is provided for you.
// REQUIRES: img points to a valid Image
// MODIFIES: *img
// EFFECTS:  The image is rotated 90 degrees to the left (counterclockwise).
void RotateLeft(Image* Img) {

  // for convenience
  int width = Image_width(Img);
  int Height = Image_height(Img);

  // auxiliary image to temporarily store rotated image
  Image *Aux = new Image;
  Image_init(Aux, Height, width); // width and height switched

  // iterate through pixels and place each where it goes in temp
  for (int r = 0; r < Height; ++r) {
    for (int c = 0; c < width; ++c) {
      Image_set_pixel(Aux, width - 1 - c, r, Image_get_pixel(Img, r, c));
    }
  }

  // Copy data back into original
  *Img = *Aux;
  delete Aux;
}
// ^ DO NOT CHANGE ^ ------------------------------------------------

// v DO NOT CHANGE v ------------------------------------------------
// The implementation of rotate_right is provided for you.
// REQUIRES: img points to a valid Image.
// MODIFIES: *img
// EFFECTS:  The image is rotated 90 degrees to the right (clockwise).
void RotateRight(Image* Img){

  // for convenience
  int width = Image_width(Img);
  int Height = Image_height(Img);

  // auxiliary image to temporarily store rotated image
  Image *Aux = new Image;
  Image_init(Aux, Height, width); // width and height switched

  // iterate through pixels and place each where it goes in temp
  for (int r = 0; r < Height; ++r) {
    for (int c = 0; c < width; ++c) {
      Image_set_pixel(Aux, c, Height - 1 - r, Image_get_pixel(Img, r, c));
    }
  }

  // Copy data back into original
  *Img = *Aux;
  delete Aux;
}
// ^ DO NOT CHANGE ^ ------------------------------------------------


// v DO NOT CHANGE v ------------------------------------------------
// The implementation of diff2 is provided for you.
static int SquaredDifference(Pixel P1, Pixel P2) {
  int Dr = P2.r - P1.r;
  int Dg = P2.g - P1.g;
  int Db = P2.b - P1.b;
  // Divide by 100 is to avoid possible overflows
  // later on in the algorithm.
  return (Dr*Dr + Dg*Dg + Db*Db) / 100;
}
// ^ DO NOT CHANGE ^ ------------------------------------------------


// ------------------------------------------------------------------
// You may change code below this line!



// REQUIRES: img points to a valid Image.
//           energy points to a Matrix.
// MODIFIES: *energy
// EFFECTS:  energy serves as an "output parameter".
//           The Matrix pointed to by energy is initialized to be the same
//           size as the given Image, and then the energy matrix for that
//           image is computed and written into it.
//           See the project spec for details on computing the energy matrix.
void ComputeEnergyMatrix(const Image* Img, Matrix* Energy) {
  Matrix_init(Energy, Img->width, Img->Height);
  Matrix_fill(Energy, 0);

  for(int i = 1; i < Img->Height - 1 ; i++) {
     for(int j = 1; j < Img->width -1 ; j++) {
       Pixel North = Image_get_pixel(Img, i-1, j);
       Pixel South = Image_get_pixel(Img, i+1, j);
       Pixel West = Image_get_pixel(Img, i, j-1);
       Pixel East = Image_get_pixel(Img, i, j+1);
       * Matrix_at(Energy, i, j) 
       = SquaredDifference(North, South) 
       + SquaredDifference(West, East);
    }
  }
  int max = Matrix_max(Energy);
  Matrix_fill_border(Energy, max);
}


// REQUIRES: energy points to a valid Matrix.
//           cost points to a Matrix.
//           energy and cost aren't pointing to the same Matrix
// MODIFIES: *cost
// EFFECTS:  cost serves as an "output parameter".
//           The Matrix pointed to by cost is initialized to be the same
//           size as the given energy Matrix, and then the cost matrix is
//           computed and written into it.
//           See the project spec for details on computing the cost matrix.
void ComputeVerticalCostMatrix(const Matrix* Energy, Matrix *Cost) {
  Matrix_init(Cost, Energy->width, Energy->Height);

  for(int i = 0; i < Cost->width; i++) {
  * Matrix_at(Cost, 0, i) = * Matrix_at(Energy, 0, i);
  }

  for(int i = 1; i < Cost->Height; i++) {
    for(int j = 0; j < Cost->width; j++) {
      if(j == 0) {
        * Matrix_at(Cost, i, j) = * Matrix_at(Energy, i, j) 
        + Matrix_min_value_in_row(Cost, i-1, j, j+2);
      }
      else if(0 < j && j < Cost->width - 1) {
        * Matrix_at(Cost, i, j) = * Matrix_at(Energy, i, j) 
        + Matrix_min_value_in_row(Cost, i-1, j-1, j+2);
      }
      else if(j == (Cost->width - 1)) {
        * Matrix_at(Cost, i, j) = * Matrix_at(Energy, i, j) 
        + Matrix_min_value_in_row(Cost, i-1, j-1, j+1);
      }
    }
  }
}


// REQUIRES: cost points to a valid Matrix
//           seam points to an array
//           the size of seam is >= Matrix_height(cost)
// MODIFIES: seam[0]...seam[Matrix_height(cost)-1]
// EFFECTS:  seam serves as an "output parameter".
//           The vertical seam with the minimal cost according to the given
//           cost matrix is found and the seam array is filled with the column
//           numbers for each pixel along the seam, with the pixel for each
//           row r placed at seam[r]. While determining the seam, if any pixels
//           tie for lowest cost, the leftmost one (i.e. with the lowest
//           column number) is used.
//           See the project spec for details on computing the minimal seam.
// NOTE:     You should compute the seam in reverse order, starting
//           with the bottom of the image and proceeding to the top,
//           as described in the project spec.
void FindMinimalVerticalSeam(const Matrix* Cost, int Seam[]) {
  int Height = Cost->Height;
  int width = Cost->width;

  Seam[Height - 1] = Matrix_column_of_min_value_in_row(Cost, Height-1, 0, width);

  for(int i = Height-2; i >= 0; i--) {
    int Cur = Seam[i+1];
    if(Cur == 0) {
      Seam[i] = Matrix_column_of_min_value_in_row(Cost, i, Cur, Cur + 2);
    }
    else if(0 < Cur && Cur < width - 1) {
      Seam[i] = Matrix_column_of_min_value_in_row(Cost, i, Cur - 1, Cur + 2);
    } 
    else if(Cur == width - 1) {
      Seam[i] = Matrix_column_of_min_value_in_row(Cost, i, Cur - 1, Cur + 1);
    }
  }
}


// REQUIRES: img points to a valid Image with width >= 2
//           seam points to an array
//           the size of seam is == Image_height(img)
//           each element x in seam satisfies 0 <= x < Image_width(img)
// MODIFIES: *img
// EFFECTS:  Removes the given vertical seam from the Image. That is, one
//           pixel will be removed from every row in the image. The pixel
//           removed from row r will be the one with column equal to seam[r].
//           The width of the image will be one less than before.
//           See the project spec for details on removing a vertical seam.
// NOTE:     Use the new operator here to create the smaller Image,
//           and then use delete when you are done with it.
void RemoveVerticalSeam(Image *Img, const int Seam[]) {
  Image* Removed = new Image;
  Image_init(Removed, Img->width-1, Img->Height);

  for(int i = 0; i < Img->Height; i++) {
    int Index = 0;
    for(int j = 0; j < Img->width; j++) {
      if(j != Seam[i]) {
        *Matrix_at(&(Removed->RedChannel), i, Index) 
        =*Matrix_at(&Img->RedChannel, i, j);
        
        *Matrix_at(&(Removed->GreenChannel), i, Index) 
        =*Matrix_at(&Img->GreenChannel, i, j);
        
        *Matrix_at(&(Removed->BlueChannel), i, Index) 
        =*Matrix_at(&Img->BlueChannel, i, j);
        Index = Index + 1;
      }
    }
  }

  *Img = *Removed;
  delete Removed;
}


// REQUIRES: img points to a valid Image
//           0 < newWidth && newWidth <= Image_width(img)
// MODIFIES: *img
// EFFECTS:  Reduces the width of the given Image to be newWidth by using
//           the seam carving algorithm. See the spec for details.
// NOTE:     Use the new operator here to create Matrix objects, and
//           then use delete when you are done with them.
void SeamCarveWidth(Image *Img, int NewWidth) {
  //assert(0 < newWidth && newWidth <= Image_width(img));

  for(int i = Img->width; i > NewWidth; i--) {
    Matrix* Energy = new Matrix;
    Matrix* Cost = new Matrix;
    int Seam[MAX_MATRIX_HEIGHT];

    ComputeEnergyMatrix(Img, Energy);
    ComputeVerticalCostMatrix(Energy, Cost);
    FindMinimalVerticalSeam(Cost, Seam);
    RemoveVerticalSeam(Img, Seam);
    
    delete Energy;
    delete Cost;
  }
}

// REQUIRES: img points to a valid Image
//           0 < newHeight && newHeight <= Image_height(img)
// MODIFIES: *img
// EFFECTS:  Reduces the height of the given Image to be newHeight.
// NOTE:     This is equivalent to first rotating the Image 90 degrees left,
//           then applying seam_carve_width(img, newHeight), then rotating
//           90 degrees right.
void SeamCarveHeight(Image *Img, int NewHeight) {
  //assert(0 < newHeight && newHeight <= Image_height(img));

  RotateLeft(Img);
  SeamCarveWidth(Img, NewHeight);
  RotateRight(Img);
}

// REQUIRES: img points to a valid Image
//           0 < newWidth && newWidth <= Image_width(img)
//           0 < newHeight && newHeight <= Image_height(img)
// MODIFIES: *img
// EFFECTS:  Reduces the width and height of the given Image to be newWidth
//           and newHeight, respectively.
// NOTE:     This is equivalent to applying seam_carve_width(img, newWidth)
//           and then applying seam_carve_height(img, newHeight).
void SeamCarve(Image *Img, int NewWidth, int NewHeight) {
  //assert(0 < newWidth && newWidth <= Image_width(img));
  //assert(0 < newHeight && newHeight <= Image_height(img));

  SeamCarveWidth(Img, NewWidth);
  SeamCarveHeight(Img, NewHeight);
}
