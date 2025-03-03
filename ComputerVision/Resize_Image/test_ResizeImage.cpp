/**
 * @file test_ResizeImage.cpp
 * @brief Unit tests for ResizeImage seam carving module
 */

#include "Matrix.h"
#include "Image.h"
#include "processing.h"
#include <cassert>
#include <iostream>
#include <sstream>

namespace {

void TestMatrixInit() {
    Matrix m;
    Matrix_init(&m, 5, 3);
    assert(Matrix_width(&m) == 5);
    assert(Matrix_height(&m) == 3);
    std::cout << "  TestMatrixInit PASSED\n";
}

void TestMatrixFill() {
    Matrix m;
    Matrix_init(&m, 4, 4);
    Matrix_fill(&m, 42);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            assert(*Matrix_at(&m, r, c) == 42);
    std::cout << "  TestMatrixFill PASSED\n";
}

void TestMatrixMax() {
    Matrix m;
    Matrix_init(&m, 3, 3);
    Matrix_fill(&m, 10);
    *Matrix_at(&m, 1, 1) = 99;
    assert(Matrix_max(&m) == 99);
    std::cout << "  TestMatrixMax PASSED\n";
}

void TestMatrixMinInRow() {
    Matrix m;
    Matrix_init(&m, 5, 1);
    int vals[] = {5, 3, 1, 4, 2};
    for (int c = 0; c < 5; ++c) *Matrix_at(&m, 0, c) = vals[c];
    assert(Matrix_min_value_in_row(&m, 0, 0, 5) == 1);
    assert(Matrix_column_of_min_value_in_row(&m, 0, 0, 5) == 2);
    std::cout << "  TestMatrixMinInRow PASSED\n";
}

void TestImageInitDimensions() {
    Image img;
    Image_init(&img, 10, 8);
    assert(Image_width(&img) == 10);
    assert(Image_height(&img) == 8);
    std::cout << "  TestImageInitDimensions PASSED\n";
}

void TestImagePixelAccess() {
    Image img;
    Image_init(&img, 5, 5);
    Pixel px = {100, 150, 200};
    Image_set_pixel(&img, 2, 3, px);
    Pixel got = Image_get_pixel(&img, 2, 3);
    assert(got.r == 100 && got.g == 150 && got.b == 200);
    std::cout << "  TestImagePixelAccess PASSED\n";
}

void TestImageFill() {
    Image img;
    Image_init(&img, 3, 3);
    Pixel color = {10, 20, 30};
    Image_fill(&img, color);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            Pixel p = Image_get_pixel(&img, r, c);
            assert(p.r == 10 && p.g == 20 && p.b == 30);
        }
    std::cout << "  TestImageFill PASSED\n";
}

void TestSeamCarveReducesWidth() {
    Image img;
    Image_init(&img, 5, 3);
    Pixel white = {255, 255, 255};
    Image_fill(&img, white);
    // Set a vertical seam of dark pixels
    for (int r = 0; r < 3; ++r)
        Image_set_pixel(&img, r, 2, {0, 0, 0});

    SeamCarveWidth(&img, 4);
    assert(Image_width(&img) == 4);
    std::cout << "  TestSeamCarveReducesWidth PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running ResizeImage tests...\n";
    TestMatrixInit();
    TestMatrixFill();
    TestMatrixMax();
    TestMatrixMinInRow();
    TestImageInitDimensions();
    TestImagePixelAccess();
    TestImageFill();
    TestSeamCarveReducesWidth();
    std::cout << "All ResizeImage tests PASSED.\n";
    return 0;
}
