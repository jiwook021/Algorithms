#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

#include "LinkedList.h"

typedef struct _sparseMatrix
{
    LinkedList* WList_;
    LinkedList* HList_;
} SparseMatrix;

void InitMatrix(SparseMatrix* Mat, int Height, int Width);
void MatInsert(SparseMatrix* Mat, char DataVal, int Row, int Col);
void PrintMat(SparseMatrix* Mat, int Row, int Col);
void PrintRow(SparseMatrix* Mat, int Row);
void PrintCol(SparseMatrix* Mat, int Col);
void MatDelete(SparseMatrix* Mat, char DataVal, int Row);
void PrintMatrix(SparseMatrix* Mat, int Row, int Col);

#endif