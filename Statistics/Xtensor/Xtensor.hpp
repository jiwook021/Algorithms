/**
 * @file Xtensor.hpp
 * @brief Minimal N-dimensional array library inspired by xtensor.
 *
 * @details
 * Provides a dynamic, N-dimensional array class (Xarray) with row-major
 * storage and common creation functions: Zeros, Ones, Linspace, Arange,
 * Eye, Reshape, Transpose, Flip.
 */

#ifndef XTENSOR_HPP
#define XTENSOR_HPP

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace Xt {

/**
 * @class Xarray
 * @brief Dynamic N-dimensional array with row-major storage.
 * @tparam T Element type.
 */
template <typename T>
class Xarray {
public:
    std::vector<size_t> Shape;    ///< Shape of the array (e.g., {3,4})
    std::vector<size_t> Strides;  ///< Row-major strides
    std::vector<T> Data;          ///< Contiguous element storage

    /** @brief Default constructor. */
    Xarray() = default;

    /**
     * @brief Construct an array with a given shape (elements default-initialized).
     * @param shp Shape of the array.
     */
    explicit Xarray(const std::vector<size_t>& shp) : Shape(shp) {
        ComputeStrides();
        Data.resize(Size());
    }

    /** @brief Compute row-major strides from the current shape. */
    void ComputeStrides() {
        Strides.resize(Shape.size());
        if (Shape.empty()) return;
        Strides.back() = 1;
        for (int i = static_cast<int>(Shape.size()) - 2; i >= 0; --i) {
            Strides[i] = Strides[i + 1] * Shape[i + 1];
        }
    }

    /** @brief Total number of elements. */
    size_t Size() const {
        size_t s = 1;
        for (auto d : Shape) s *= d;
        return s;
    }

    /**
     * @brief Element access via initializer list of indices.
     * @param indices Multi-dimensional indices.
     * @return Reference to the element.
     */
    T& operator()(std::initializer_list<size_t> indices) {
        assert(indices.size() == Shape.size() && "Incorrect number of indices");
        size_t idx = 0;
        size_t i = 0;
        for (auto ind : indices) {
            assert(ind < Shape[i] && "Index out of bounds");
            idx += ind * Strides[i];
            ++i;
        }
        return Data[idx];
    }

    /** @brief Const element access via initializer list. */
    const T& operator()(std::initializer_list<size_t> indices) const {
        assert(indices.size() == Shape.size() && "Incorrect number of indices");
        size_t idx = 0;
        size_t i = 0;
        for (auto ind : indices) {
            assert(ind < Shape[i] && "Index out of bounds");
            idx += ind * Strides[i];
            ++i;
        }
        return Data[idx];
    }
};

/**
 * @brief Create an array filled with zeros.
 * @param shape Shape of the array.
 */
template <typename T>
Xarray<T> Zeros(const std::vector<size_t>& shape) {
    Xarray<T> arr(shape);
    std::fill(arr.Data.begin(), arr.Data.end(), T(0));
    return arr;
}

/**
 * @brief Create an array filled with ones.
 * @param shape Shape of the array.
 */
template <typename T>
Xarray<T> Ones(const std::vector<size_t>& shape) {
    Xarray<T> arr(shape);
    std::fill(arr.Data.begin(), arr.Data.end(), T(1));
    return arr;
}

/**
 * @brief Create a 1D array of evenly spaced values.
 * @param start Start value (inclusive).
 * @param stop Stop value (inclusive).
 * @param num Number of values.
 */
template <typename T>
Xarray<T> Linspace(T start, T stop, size_t num) {
    Xarray<T> arr({num});
    if (num == 0) return arr;
    if (num == 1) {
        arr.Data[0] = start;
        return arr;
    }
    T step = (stop - start) / static_cast<T>(num - 1);
    for (size_t i = 0; i < num; ++i) {
        arr.Data[i] = start + step * static_cast<T>(i);
    }
    arr.Data[num - 1] = stop;  // Ensure exact endpoint
    return arr;
}

/**
 * @brief Create a 1D array with values from start to stop (exclusive).
 * @param start Start value (inclusive).
 * @param stop Stop value (exclusive).
 * @param step Step size.
 */
template <typename T>
Xarray<T> Arange(T start, T stop, T step = T(1)) {
    assert(step != T(0) && "Step must be non-zero");
    std::vector<T> values;
    if (step > T(0)) {
        for (T v = start; v < stop; v += step) values.push_back(v);
    } else {
        for (T v = start; v > stop; v += step) values.push_back(v);
    }
    Xarray<T> arr({values.size()});
    arr.Data = values;
    return arr;
}

/**
 * @brief Create an NxN identity matrix.
 * @param n Size of the identity matrix.
 */
template <typename T>
Xarray<T> Eye(size_t n) {
    Xarray<T> arr({n, n});
    std::fill(arr.Data.begin(), arr.Data.end(), T(0));
    for (size_t i = 0; i < n; ++i) {
        arr({i, i}) = T(1);
    }
    return arr;
}

/**
 * @brief Reshape an array to a new shape (total size must match).
 */
template <typename T>
Xarray<T> Reshape(const Xarray<T>& arr, const std::vector<size_t>& newShape) {
    size_t newTotal = 1;
    for (auto s : newShape) newTotal *= s;
    assert(newTotal == arr.Size() && "Total size must remain the same");
    Xarray<T> result(newShape);
    result.Data = arr.Data;
    return result;
}

/**
 * @brief Transpose a 2D array (swap rows and columns).
 */
template <typename T>
Xarray<T> Transpose(const Xarray<T>& arr) {
    assert(arr.Shape.size() == 2 && "Transpose only for 2D arrays");
    std::vector<size_t> newShape = {arr.Shape[1], arr.Shape[0]};
    Xarray<T> result(newShape);
    for (size_t i = 0; i < arr.Shape[0]; ++i) {
        for (size_t j = 0; j < arr.Shape[1]; ++j) {
            result({j, i}) = arr({i, j});
        }
    }
    return result;
}

/**
 * @brief Flip an array along a given axis.
 */
template <typename T>
Xarray<T> Flip(const Xarray<T>& arr, size_t axis) {
    assert(axis < arr.Shape.size() && "Axis out of bounds");
    Xarray<T> result(arr.Shape);
    auto computeMultiIndex = [&](size_t linIdx) -> std::vector<size_t> {
        std::vector<size_t> idx(arr.Shape.size());
        for (size_t i = 0; i < arr.Shape.size(); ++i) {
            idx[i] = (linIdx / arr.Strides[i]) % arr.Shape[i];
        }
        return idx;
    };
    for (size_t lin = 0; lin < arr.Size(); ++lin) {
        auto idx = computeMultiIndex(lin);
        idx[axis] = arr.Shape[axis] - 1 - idx[axis];
        size_t newLin = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            newLin += idx[i] * result.Strides[i];
        }
        result.Data[newLin] = arr.Data[lin];
    }
    return result;
}

}  // namespace Xt

#endif // XTENSOR_HPP
