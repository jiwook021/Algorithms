/**
 * @file main.cpp
 * @brief Driver for SparseMatrix (orthogonal linked-list representation)
 */

#include "SparseMatrix.hpp"
#include <iostream>

int main() {
    SparseMatrix mat(9, 9);

    mat.Insert(1, 2, 'a');
    mat.Insert(2, 3, 'b');
    mat.Insert(1, 3, 'c');
    mat.Insert(2, 2, 'd');
    mat.Insert(2, 9, 'e');
    mat.Insert(9, 2, 'f');

    std::cout << "Sparse Matrix (9x9):\n";
    mat.Print();

    std::cout << "\nNon-zero count: " << mat.Count() << "\n";

    std::cout << "\nRow 2 entries: ";
    for (auto [col, val] : mat.GetRow(2)) {
        std::cout << "(" << col << "," << val << ") ";
    }
    std::cout << "\n";

    std::cout << "Col 2 entries: ";
    for (auto [row, val] : mat.GetCol(2)) {
        std::cout << "(" << row << "," << val << ") ";
    }
    std::cout << "\n";

    mat.Remove(2, 3);
    std::cout << "\nAfter removing (2,3):\n";
    mat.Print();

    return 0;
}
