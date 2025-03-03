#include "SparseMatrix.h"
#include "LinkedList.h"
#include <stdio.h>
#include <stdlib.h>

void InitMatrix(SparseMatrix* Mat, int Height, int Width)
{
    Mat->WList_ = (LinkedList*)malloc(sizeof(LinkedList) * Width);
    if (Mat->WList_ == NULL) return;

    for (int I = 0; I < Width; I++)
    {
        InitList((Mat->WList_) + I);
    }
    Mat->HList_ = (LinkedList*)malloc(sizeof(LinkedList) * Height);
    if (Mat->HList_ == NULL) { free(Mat->WList_); Mat->WList_ = NULL; return; }
    for (int I = 0; I < Height; I++)
    {
        InitList((Mat->HList_) + I);
    }
}


void MatInsert(SparseMatrix* Mat, char DataVal, int Row, int Col)
{
    Node* NewNode = (Node*)malloc(sizeof(Node));
    if (NewNode == NULL) return;
    LNInsert(&(Mat->HList_[Row]), NewNode, DataVal);
    LBInsert(&(Mat->WList_[Col]), NewNode, DataVal);
}



void MatDelete(SparseMatrix* Mat, char DataVal, int Row)
{
    for (int I = 0; I <= Row; I++)
    {
        NodeDelete(&(Mat->HList_[I]), DataVal);
    }
}



void PrintMatrix(SparseMatrix* Mat, int Row, int Col)
{
    Mat->HList_[Row].Current_ = Mat->HList_[Row].Head_;

    printf("Matrix\n\n");

    for (int J = 1; J <= Row; J++)
    {
        printf("\nrow %d: ", J);

        for (int I = 0; I < Mat->HList_[J].Size_; I++)
        {
            printf("%c ", Mat->HList_[J].Current_->Data_);
            Mat->HList_[J].Current_ = Mat->HList_[J].Current_->Next_;
        }
    }
    printf("\n");
}



void PrintMat(SparseMatrix* Mat, int Row, int Col)
{
    for (int I = 1; I <= Row; I++)
    {
        if (Mat->HList_[I].Size_ == 0) continue;
        
        PrintRow(Mat, I);

    }
    printf("\n\n");
    for (int J = 1; J <= Col; J++)
    {

        if (Mat->WList_[J].Size_ == 0) continue;

        PrintCol(Mat, J);
    }
}


void PrintRow(SparseMatrix* Mat, int Row)
{
    Mat->HList_[Row].Current_ = Mat->HList_[Row].Head_;

    printf("Matrix row %d: ", Row);

    for (int I = 0; I < Mat->HList_[Row].Size_; I++)
    {
        printf("%c ", Mat->HList_[Row].Current_->Data_);
        Mat->HList_[Row].Current_ = Mat->HList_[Row].Current_->Next_;
    }
    printf("\n");
}

void PrintCol(SparseMatrix* Mat, int Col)
{
    Mat->WList_[Col].Current_ = Mat->WList_[Col].Head_;

    printf("Matrix column %d: ", Col);

    for (int I = 0; I < Mat->WList_[Col].Size_; I++)
    {
        printf("%c ", Mat->WList_[Col].Current_->Data_);
        Mat->WList_[Col].Current_ = Mat->WList_[Col].Current_->Below_;
    }

    printf("\n");
}
