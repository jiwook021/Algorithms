/**
 * @file SegmentTree.cpp
 * @brief Implementation of SegmentTree<T>.
 */

#include "SegmentTree.hpp"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename T>
SegmentTree<T>::SegmentTree(const std::vector<T>& data,
                            T identity,
                            std::function<T(T, T)> op)
    : size_(0), identity_(identity), op_(std::move(op)) {
    Build(data);
}

// ---------------------------------------------------------------------------
// Build -- bottom-up tree construction
// ---------------------------------------------------------------------------

template <typename T>
void SegmentTree<T>::Build(const std::vector<T>& data) {
    std::size_t n = data.size();
    size_ = 1;
    while (size_ < n) size_ *= 2;

    tree_.assign(2 * size_, identity_);

    // Copy data into leaf positions [size_ .. size_+n-1].
    for (std::size_t i = 0; i < n; ++i) {
        tree_[size_ + i] = data[i];
    }

    // Fill internal nodes from bottom to top.
    for (std::size_t i = size_ - 1; i >= 1; --i) {
        tree_[i] = op_(tree_[i * 2], tree_[i * 2 + 1]);
    }
}

// ---------------------------------------------------------------------------
// Update -- point update with leaf-to-root propagation
// ---------------------------------------------------------------------------

template <typename T>
void SegmentTree<T>::Update(std::size_t index, T value) {
    std::size_t pos = index + size_;   // leaf position in tree_
    tree_[pos] = value;
    for (pos /= 2; pos >= 1; pos /= 2) {
        tree_[pos] = op_(tree_[pos * 2], tree_[pos * 2 + 1]);
    }
}

// ---------------------------------------------------------------------------
// Query -- range query delegating to recursive helper
// ---------------------------------------------------------------------------

template <typename T>
T SegmentTree<T>::Query(std::size_t left, std::size_t right) const {
    return QueryHelper(1, 0, size_ - 1, left, right);
}

// ---------------------------------------------------------------------------
// QueryHelper -- recursive three-case range decomposition
// ---------------------------------------------------------------------------

template <typename T>
T SegmentTree<T>::QueryHelper(std::size_t node, std::size_t start,
                              std::size_t end, std::size_t left,
                              std::size_t right) const {
    if (start > right || end < left) return identity_;
    if (left <= start && right >= end) return tree_[node];
    std::size_t mid = (start + end) / 2;
    return op_(QueryHelper(node * 2, start, mid, left, right),
               QueryHelper(node * 2 + 1, mid + 1, end, left, right));
}

// ---------------------------------------------------------------------------
// Size
// ---------------------------------------------------------------------------

template <typename T>
std::size_t SegmentTree<T>::Size() const {
    return size_;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class SegmentTree<int>;
template class SegmentTree<double>;
template class SegmentTree<long long>;
