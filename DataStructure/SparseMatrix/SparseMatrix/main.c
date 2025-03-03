#include <stdio.h>
#include <stdlib.h>
#include "SparseMatrix.h"
#include "LinkedList.h"

int main()
{
    SparseMatrix Mat;
    InitMatrix(&Mat, 100, 100);

    MatInsert(&Mat, 'a', 1, 2);
    MatInsert(&Mat, 'b', 2, 3);
    MatInsert(&Mat, 'c', 1, 3);
    MatInsert(&Mat, 'd', 2, 2);
    MatInsert(&Mat, 'e', 2, 9);
    MatInsert(&Mat, 'f', 9, 2);

    PrintMatrix(&Mat, 9, 9);
    return 0;
}