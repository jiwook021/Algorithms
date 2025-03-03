/**
 * @file test_SparseMatrix.cpp
 * @brief Unit tests for SparseMatrix (orthogonal linked-list sparse matrix)
 */

#include "SparseMatrix.hpp"
#include <gtest/gtest.h>

TEST(SparseMatrixTest, Construction) {
    SparseMatrix mat(5, 5);
    EXPECT_EQ(mat.Rows(), 5u);
    EXPECT_EQ(mat.Cols(), 5u);
    EXPECT_EQ(mat.Count(), 0u);
    EXPECT_TRUE(mat.Empty());
}

TEST(SparseMatrixTest, InvalidConstruction) {
    EXPECT_THROW(SparseMatrix(0, 5), std::invalid_argument);
    EXPECT_THROW(SparseMatrix(5, 0), std::invalid_argument);
}

TEST(SparseMatrixTest, InsertAndGet) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 2, 'a');
    mat.Insert(3, 4, 'b');

    EXPECT_EQ(mat.Get(1, 2), 'a');
    EXPECT_EQ(mat.Get(3, 4), 'b');
    EXPECT_EQ(mat.Get(1, 1), '\0');  // empty cell
    EXPECT_EQ(mat.Count(), 2u);
}

TEST(SparseMatrixTest, InsertOverwrite) {
    SparseMatrix mat(5, 5);
    mat.Insert(2, 3, 'x');
    mat.Insert(2, 3, 'y');

    EXPECT_EQ(mat.Get(2, 3), 'y');
    EXPECT_EQ(mat.Count(), 1u);  // still only one entry
}

TEST(SparseMatrixTest, Contains) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 1, 'a');

    EXPECT_TRUE(mat.Contains(1, 1));
    EXPECT_FALSE(mat.Contains(1, 2));
}

TEST(SparseMatrixTest, Remove) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 2, 'a');
    mat.Insert(2, 3, 'b');

    EXPECT_TRUE(mat.Remove(1, 2));
    EXPECT_FALSE(mat.Contains(1, 2));
    EXPECT_EQ(mat.Count(), 1u);

    EXPECT_FALSE(mat.Remove(1, 2));  // already removed
    EXPECT_FALSE(mat.Remove(4, 4));  // never existed
}

TEST(SparseMatrixTest, GetRow) {
    SparseMatrix mat(5, 5);
    mat.Insert(2, 1, 'a');
    mat.Insert(2, 4, 'b');
    mat.Insert(2, 2, 'c');

    auto row = mat.GetRow(2);
    ASSERT_EQ(row.size(), 3u);
    // Should be sorted by column
    EXPECT_EQ(row[0].first, 1u);
    EXPECT_EQ(row[0].second, 'a');
    EXPECT_EQ(row[1].first, 2u);
    EXPECT_EQ(row[1].second, 'c');
    EXPECT_EQ(row[2].first, 4u);
    EXPECT_EQ(row[2].second, 'b');
}

TEST(SparseMatrixTest, GetCol) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 3, 'x');
    mat.Insert(4, 3, 'y');
    mat.Insert(2, 3, 'z');

    auto col = mat.GetCol(3);
    ASSERT_EQ(col.size(), 3u);
    // Should be sorted by row
    EXPECT_EQ(col[0].first, 1u);
    EXPECT_EQ(col[0].second, 'x');
    EXPECT_EQ(col[1].first, 2u);
    EXPECT_EQ(col[1].second, 'z');
    EXPECT_EQ(col[2].first, 4u);
    EXPECT_EQ(col[2].second, 'y');
}

TEST(SparseMatrixTest, EmptyRow) {
    SparseMatrix mat(5, 5);
    auto row = mat.GetRow(3);
    EXPECT_TRUE(row.empty());
}

TEST(SparseMatrixTest, Clear) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 1, 'a');
    mat.Insert(2, 2, 'b');
    mat.Insert(3, 3, 'c');

    mat.Clear();
    EXPECT_TRUE(mat.Empty());
    EXPECT_EQ(mat.Count(), 0u);
    EXPECT_FALSE(mat.Contains(1, 1));

    // Can insert again after clear
    mat.Insert(1, 1, 'z');
    EXPECT_EQ(mat.Get(1, 1), 'z');
}

TEST(SparseMatrixTest, OutOfRange) {
    SparseMatrix mat(5, 5);
    EXPECT_THROW(mat.Insert(0, 1, 'a'), std::out_of_range);
    EXPECT_THROW(mat.Insert(6, 1, 'a'), std::out_of_range);
    EXPECT_THROW(mat.Insert(1, 6, 'a'), std::out_of_range);
    EXPECT_THROW(mat.Get(0, 0), std::out_of_range);
}

TEST(SparseMatrixTest, MoveConstruction) {
    SparseMatrix mat(5, 5);
    mat.Insert(1, 1, 'a');
    mat.Insert(2, 2, 'b');

    SparseMatrix mat2(std::move(mat));
    EXPECT_EQ(mat2.Count(), 2u);
    EXPECT_EQ(mat2.Get(1, 1), 'a');
    EXPECT_EQ(mat2.Get(2, 2), 'b');
}

TEST(SparseMatrixTest, MultipleInsertionsAndRemovals) {
    SparseMatrix mat(10, 10);
    for (std::size_t i = 1; i <= 10; ++i) {
        mat.Insert(i, i, static_cast<char>('a' + i - 1));
    }
    EXPECT_EQ(mat.Count(), 10u);

    for (std::size_t i = 1; i <= 5; ++i) {
        mat.Remove(i, i);
    }
    EXPECT_EQ(mat.Count(), 5u);

    for (std::size_t i = 1; i <= 5; ++i) {
        EXPECT_FALSE(mat.Contains(i, i));
    }
    for (std::size_t i = 6; i <= 10; ++i) {
        EXPECT_TRUE(mat.Contains(i, i));
    }
}
