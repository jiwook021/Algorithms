/**
 * @file SparseMatrix.hpp
 * @brief Sparse matrix using cross-linked lists (orthogonal list representation).
 * @details Stores only non-zero elements in linked lists organized by row and column.
 *          Each non-zero entry is linked into both its row chain and its column chain,
 *          allowing efficient row/column traversal. Memory usage is O(k) where k is
 *          the number of non-zero entries, compared to O(m*n) for a dense matrix.
 *
 * Complexity:
 *   Insert  : O(r + c) where r = non-zeros in row, c = non-zeros in column
 *   Remove  : O(r + c)
 *   Get     : O(r) or O(c)
 *   Row/Col traversal : O(non-zeros in that row/col)
 *   Space   : O(k + m + n) where k = non-zero count, m = rows, n = cols
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * @class SparseMatrix
 * @brief Orthogonal linked-list sparse matrix for char data.
 *
 * Uses cross-linked nodes where each node belongs to both a row list
 * and a column list, enabling efficient access in either dimension.
 */
class SparseMatrix {
public:
    /**
     * @brief Construct a sparse matrix with the given dimensions.
     * @param rows Number of rows (1-indexed: valid rows are 1..rows).
     * @param cols Number of columns (1-indexed: valid cols are 1..cols).
     * @throws std::invalid_argument if rows or cols is 0.
     */
    SparseMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), count_(0) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("SparseMatrix: dimensions must be > 0");
        }
        rowHeads_.resize(rows + 1, nullptr);
        colHeads_.resize(cols + 1, nullptr);
    }

    /** @brief Destructor frees all nodes. */
    ~SparseMatrix() noexcept {
        Clear();
    }

    /// Non-copyable for simplicity (raw pointers).
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;

    /// Move constructor.
    SparseMatrix(SparseMatrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), count_(other.count_),
          rowHeads_(std::move(other.rowHeads_)),
          colHeads_(std::move(other.colHeads_)) {
        other.count_ = 0;
    }

    /// Move assignment.
    SparseMatrix& operator=(SparseMatrix&& other) noexcept {
        if (this != &other) {
            Clear();
            rows_ = other.rows_;
            cols_ = other.cols_;
            count_ = other.count_;
            rowHeads_ = std::move(other.rowHeads_);
            colHeads_ = std::move(other.colHeads_);
            other.count_ = 0;
        }
        return *this;
    }

    /**
     * @brief Insert a non-zero element at (row, col).
     * @param row 1-based row index.
     * @param col 1-based column index.
     * @param value Character data to store.
     * @throws std::out_of_range if row or col is out of bounds.
     *
     * If an element already exists at (row, col), its value is overwritten.
     */
    void Insert(std::size_t row, std::size_t col, char value) {
        ValidateIndices(row, col);

        // Check if element already exists
        Node* existing = FindNode(row, col);
        if (existing) {
            existing->data = value;
            return;
        }

        Node* node = new Node{value, row, col, nullptr, nullptr};

        // Insert into row list (sorted by column)
        InsertIntoRowList(node);
        // Insert into column list (sorted by row)
        InsertIntoColList(node);

        ++count_;
    }

    /**
     * @brief Remove the element at (row, col).
     * @return true if an element was removed, false if position was empty.
     */
    bool Remove(std::size_t row, std::size_t col) {
        ValidateIndices(row, col);

        // Remove from row list
        Node* prev = nullptr;
        Node* cur = rowHeads_[row];
        while (cur && cur->col < col) {
            prev = cur;
            cur = cur->nextInRow;
        }
        if (!cur || cur->col != col) return false;

        if (prev) prev->nextInRow = cur->nextInRow;
        else rowHeads_[row] = cur->nextInRow;

        // Remove from col list
        prev = nullptr;
        Node* c = colHeads_[col];
        while (c && c->row < row) {
            prev = c;
            c = c->nextInCol;
        }
        if (prev) prev->nextInCol = cur->nextInCol;
        else colHeads_[col] = cur->nextInCol;

        delete cur;
        --count_;
        return true;
    }

    /**
     * @brief Get the value at (row, col).
     * @return The character value, or '\\0' if the position is empty.
     */
    char Get(std::size_t row, std::size_t col) const {
        ValidateIndices(row, col);
        Node* node = FindNode(row, col);
        return node ? node->data : '\0';
    }

    /**
     * @brief Check if a non-zero entry exists at (row, col).
     */
    bool Contains(std::size_t row, std::size_t col) const {
        ValidateIndices(row, col);
        return FindNode(row, col) != nullptr;
    }

    /**
     * @brief Get all non-zero values in the given row, in column order.
     * @param row 1-based row index.
     * @return Vector of (column, value) pairs.
     */
    std::vector<std::pair<std::size_t, char>> GetRow(std::size_t row) const {
        if (row == 0 || row > rows_) {
            throw std::out_of_range("SparseMatrix: row out of range");
        }
        std::vector<std::pair<std::size_t, char>> result;
        Node* cur = rowHeads_[row];
        while (cur) {
            result.emplace_back(cur->col, cur->data);
            cur = cur->nextInRow;
        }
        return result;
    }

    /**
     * @brief Get all non-zero values in the given column, in row order.
     * @param col 1-based column index.
     * @return Vector of (row, value) pairs.
     */
    std::vector<std::pair<std::size_t, char>> GetCol(std::size_t col) const {
        if (col == 0 || col > cols_) {
            throw std::out_of_range("SparseMatrix: col out of range");
        }
        std::vector<std::pair<std::size_t, char>> result;
        Node* cur = colHeads_[col];
        while (cur) {
            result.emplace_back(cur->row, cur->data);
            cur = cur->nextInCol;
        }
        return result;
    }

    /** @brief Remove all elements. */
    void Clear() noexcept {
        for (std::size_t r = 0; r < rowHeads_.size(); ++r) {
            Node* cur = rowHeads_[r];
            while (cur) {
                Node* next = cur->nextInRow;
                delete cur;
                cur = next;
            }
            rowHeads_[r] = nullptr;
        }
        for (std::size_t c = 0; c < colHeads_.size(); ++c) {
            colHeads_[c] = nullptr;
        }
        count_ = 0;
    }

    /**
     * @brief Print the matrix in dense format (for debugging small matrices).
     */
    void Print(std::ostream& os = std::cout) const {
        for (std::size_t r = 1; r <= rows_; ++r) {
            Node* cur = rowHeads_[r];
            for (std::size_t c = 1; c <= cols_; ++c) {
                if (cur && cur->col == c) {
                    os << cur->data << ' ';
                    cur = cur->nextInRow;
                } else {
                    os << ". ";
                }
            }
            os << '\n';
        }
    }

    /** @brief Return the number of non-zero elements. */
    [[nodiscard]] std::size_t Count() const noexcept { return count_; }

    /** @brief Return the number of rows. */
    [[nodiscard]] std::size_t Rows() const noexcept { return rows_; }

    /** @brief Return the number of columns. */
    [[nodiscard]] std::size_t Cols() const noexcept { return cols_; }

    /** @brief Check if the matrix has no non-zero elements. */
    [[nodiscard]] bool Empty() const noexcept { return count_ == 0; }

private:
    /** @brief Internal node storing one non-zero entry. */
    struct Node {
        char data;
        std::size_t row;
        std::size_t col;
        Node* nextInRow;  ///< Next node in the same row (sorted by col)
        Node* nextInCol;  ///< Next node in the same column (sorted by row)
    };

    std::size_t rows_;
    std::size_t cols_;
    std::size_t count_;
    std::vector<Node*> rowHeads_;  ///< Head of each row's linked list (1-indexed)
    std::vector<Node*> colHeads_;  ///< Head of each column's linked list (1-indexed)

    /** @brief Validate row/col indices. */
    void ValidateIndices(std::size_t row, std::size_t col) const {
        if (row == 0 || row > rows_ || col == 0 || col > cols_) {
            throw std::out_of_range("SparseMatrix: index out of range");
        }
    }

    /** @brief Find node at (row, col) or return nullptr. */
    Node* FindNode(std::size_t row, std::size_t col) const {
        Node* cur = rowHeads_[row];
        while (cur && cur->col < col) {
            cur = cur->nextInRow;
        }
        if (cur && cur->col == col) return cur;
        return nullptr;
    }

    /** @brief Insert node into its row's linked list (sorted by column). */
    void InsertIntoRowList(Node* node) {
        Node* prev = nullptr;
        Node* cur = rowHeads_[node->row];
        while (cur && cur->col < node->col) {
            prev = cur;
            cur = cur->nextInRow;
        }
        node->nextInRow = cur;
        if (prev) prev->nextInRow = node;
        else rowHeads_[node->row] = node;
    }

    /** @brief Insert node into its column's linked list (sorted by row). */
    void InsertIntoColList(Node* node) {
        Node* prev = nullptr;
        Node* cur = colHeads_[node->col];
        while (cur && cur->row < node->row) {
            prev = cur;
            cur = cur->nextInCol;
        }
        node->nextInCol = cur;
        if (prev) prev->nextInCol = node;
        else colHeads_[node->col] = node;
    }
};
