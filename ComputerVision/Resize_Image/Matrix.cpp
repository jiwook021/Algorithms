// Project UID af1f95f547e44c8ea88730dfb185559d

#include "Matrix.h"
#include <cassert>
// allowed?
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <string>
// REQUIRES: mat points to a Matrix
//           0 < width && width <= MAX_MATRIX_WIDTH
//           0 < height && height <= MAX_MATRIX_HEIGHT
// MODIFIES: *mat
// EFFECTS:  Initializes *mat as a Matrix with the given width and height.
// NOTE:     Do NOT use new or delete here.
void Matrix_init(Matrix* mat, int width, int Height) {
  assert(0 < width && width <= MAX_MATRIX_WIDTH);
  assert(0 < Height && Height <= MAX_MATRIX_HEIGHT);
  mat -> width = width;
  mat -> Height = Height;
}

// REQUIRES: mat points to a valid Matrix
// MODIFIES: os
// EFFECTS:  First, prints the width and height for the Matrix to os:
//             WIDTH [space] HEIGHT [newline]
//           Then prints the rows of the Matrix to os with one row per line.
//           Each element is followed by a space and each row is followed
//           by a newline. This means there will be an "extra" space at
//           the end of each line.
void Matrix_print(const Matrix* mat, std::ostream& Os) {
  Os << mat -> width << " " << mat -> Height << "\n";
  for(int i =0; i < mat -> Height; i++){
    for(int j=0; j <mat -> width; j++){
      Os << mat -> data [mat -> width * i +j] << " ";
    }
    Os << "\n";
  }
}

// REQUIRES: mat points to an valid Matrix
// EFFECTS:  Returns the width of the Matrix.
int Matrix_width(const Matrix* mat) {
  return mat-> width;
}

// REQUIRES: mat points to a valid Matrix
// EFFECTS:  Returns the height of the Matrix.
int Matrix_height(const Matrix* mat) {
  return mat -> Height;
}

// REQUIRES: mat points to a valid Matrix
//           ptr points to an element in the Matrix
// EFFECTS:  Returns the row of the element pointed to by ptr.
int Matrix_row(const Matrix* mat, const int* Ptr) {
  int Row = (int)(Ptr - mat -> data) / mat -> width;
  return Row;
}

// REQUIRES: mat points to a valid Matrix
//           ptr point to an element in the Matrix
// EFFECTS:  Returns the column of the element pointed to by ptr.
int Matrix_column(const Matrix* mat, const int* Ptr) {
  int Column = (int)(Ptr - mat -> data) % mat -> width;
  return Column;
}

// REQUIRES: mat points to a valid Matrix
//           0 <= row && row < Matrix_height(mat)
//           0 <= column && column < Matrix_width(mat)
//
// MODIFIES: (The returned pointer may be used to modify an
//            element in the Matrix.)
// EFFECTS:  Returns a pointer to the element in the Matrix
//           at the given row and column.
int* Matrix_at(Matrix* mat, int Row, int Column) {
  assert(0 <= Row && Row < Matrix_height(mat));
  assert(0 <= Column && Column < Matrix_width(mat));
  return &(mat->data[Row*(mat->width)+ Column]);
}

// REQUIRES: mat points to a valid Matrix
//           0 <= row && row < Matrix_height(mat)
//           0 <= column && column < Matrix_width(mat)
//
// EFFECTS:  Returns a pointer-to-const to the element in
//           the Matrix at the given row and column.
const int* Matrix_at(const Matrix* mat, int Row, int Column) 
{
  assert(0 <= Row && Row < Matrix_height(mat));
  assert(0 <= Column && Column < Matrix_width(mat));
  return &(mat->data[Row*(mat->width)+ Column]);
}

// REQUIRES: mat points to a valid Matrix
// MODIFIES: *mat
// EFFECTS:  Sets each element of the Matrix to the given value.
void Matrix_fill(Matrix* mat, int value) {
  int i;
  for(i=0;i<((mat->Height)*(mat->width));i++)
  {
    mat->data[i] = value;
  }
}

// REQUIRES: mat points to a valid Matrix
// MODIFIES: *mat
// EFFECTS:  Sets each element on the border of the Matrix to
//           the given value. These are all elements in the first/last
//           row or the first/last column.
void Matrix_fill_border(Matrix* mat, int value) {
  for(int i = 0;i<((mat->Height)*(mat->width)); i++){
    //first row
    if (i<=mat->width) {
      mat -> data[i] = value;
      continue;     
    }
    //Last row
    if ((mat->width)*(mat->Height-1)<i){
      mat->data[i] = value;
      continue;     
    }
    //first column
    if (i % (mat -> width) == 0){
      mat -> data[i] = value;
      continue; 
    }
    //Last column
    if (i % (mat -> width) == ((mat -> width)-1)){
      mat -> data[i] = value;
      continue; 
    }
  }
}

// REQUIRES: mat points to a valid Matrix
// EFFECTS:  Returns the value of the maximum element in the Matrix
int Matrix_max(const Matrix* mat) {
  int max = 0;
  for(int i = 0; i < ((mat-> Height) * (mat -> width)); i++){
    if(max < mat -> data[i]){
      max = mat -> data[i];
    }
  }
  return max;
}

// REQUIRES: mat points to a valid Matrix
//           0 <= row && row < Matrix_height(mat)
//           0 <= column_start && column_end <= Matrix_width(mat)
//           column_start < column_end
// EFFECTS:  Returns the column of the element with the minimal value
//           in a particular region. The region is defined as elements
//           in the given row and between column_start (inclusive) and
//           column_end (exclusive).
//           If multiple elements are minimal, returns the column of
//           the leftmost one.
int Matrix_column_of_min_value_in_row(const Matrix* mat, int Row, int ColumnStart, int ColumnEnd) 
{
  assert(0 <= Row && Row < Matrix_height(mat));
  assert(0 <= ColumnStart && ColumnEnd <= Matrix_width(mat));
  assert(ColumnStart < ColumnEnd);
  
  int Column = ColumnStart;
  int min = mat -> data[Row * (mat -> width) + ColumnStart];

  for(int i = ColumnStart; i < ColumnEnd; i++){  
    if(min > (mat -> data[Row * (mat -> width) + i])){
      min = mat -> data[Row * (mat -> width) + i];
      Column = i; 
    }
    if(min == (mat -> data[Row * (mat -> width) + i])){
      continue;
    }
  }
  return Column; 
}

// REQUIRES: mat points to a valid Matrix
//           0 <= row && row < Matrix_height(mat)
//           0 <= column_start && column_end <= Matrix_width(mat)
//           column_start < column_end
// EFFECTS:  Returns the minimal value in a particular region. The region
//           is defined as elements in the given row and between
//           column_start (inclusive) and column_end (exclusive).
int Matrix_min_value_in_row(const Matrix* mat, int Row, int ColumnStart, int ColumnEnd) {
  assert(0 <= Row && Row < Matrix_height(mat));
  assert(0 <= ColumnStart && ColumnEnd <= Matrix_width(mat));
  assert(ColumnStart < ColumnEnd);

  int min = 100000000;
  
  for(int i = ColumnStart; i < ColumnEnd; i++){  
    if(min > (mat -> data[Row * (mat -> width) + i])){
      min = mat -> data[Row * (mat -> width) + i];
    }
    if(min == (mat -> data[Row * (mat -> width) + i])){
      continue;
    }
  }
  return min; 
}
