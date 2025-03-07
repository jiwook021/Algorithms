#include <iostream>
#include <vector>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <numeric>

namespace xt {

    // Minimal xarray: a dynamic, N-dimensional array.
    template <typename T>
    class xarray {
    public:
        std::vector<size_t> shape;    // e.g. {3,4} for a 3x4 array.
        std::vector<size_t> strides;  // computed for row‐major order.
        std::vector<T> data;          // contiguous storage.

        // Default constructor.
        xarray() = default;

        // Construct an xarray with a given shape.
        explicit xarray(const std::vector<size_t>& shp)
            : shape(shp)
        {
            size_t total = 1;
            for (auto s : shape) total *= s;
            data.resize(total);
            compute_strides();
        }

        // Compute row-major strides.
        void compute_strides() {
            strides.resize(shape.size());
            if (shape.empty()) return;
            strides.back() = 1;
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        // Total number of elements.
        size_t size() const {
            return data.size();
        }

        // Element access via an initializer_list.
        T& operator()(std::initializer_list<size_t> indices) {
            assert(indices.size() == shape.size() && "Incorrect number of indices");
            size_t idx = 0;
            size_t i = 0;
            for (auto ind : indices) {
                assert(ind < shape[i] && "Index out of bounds");
                idx += ind * strides[i];
                ++i;
            }
            return data[idx];
        }
        const T& operator()(std::initializer_list<size_t> indices) const {
            assert(indices.size() == shape.size() && "Incorrect number of indices");
            size_t idx = 0;
            size_t i = 0;
            for (auto ind : indices) {
                assert(ind < shape[i] && "Index out of bounds");
                idx += ind * strides[i];
                ++i;
            }
            return data[idx];
        }
    };

    // xt::zeros: create an array with all elements = 0.
    template <typename T>
    xarray<T> zeros(const std::vector<size_t>& shape) {
        xarray<T> arr(shape);
        std::fill(arr.data.begin(), arr.data.end(), T(0));
        return arr;
    }

    // xt::ones: create an array with all elements = 1.
    template <typename T>
    xarray<T> ones(const std::vector<size_t>& shape) {
        xarray<T> arr(shape);
        std::fill(arr.data.begin(), arr.data.end(), T(1));
        return arr;
    }

    // xt::empty: create an array without initializing its values.
    // (Note: In this simplified version, elements are default-constructed.)
    template <typename T>
    xarray<T> empty(const std::vector<size_t>& shape) {
        return xarray<T>(shape);
    }

    // xt::linspace: create a 1D array of 'num' equally spaced values from start to stop (inclusive).
    template <typename T>
    xarray<T> linspace(T start, T stop, size_t num) {
        xarray<T> arr({ num });
        if (num == 0) return arr; // though typically num > 0 is expected.
        if (num == 1) {
            arr.data[0] = start;
            return arr;
        }
        T step = (stop - start) / static_cast<T>(num - 1);
        for (size_t i = 0; i < num; i++) {
            arr.data[i] = start + step * static_cast<T>(i);
        }
        arr.data[num - 1] = stop;  // ensure exact endpoint
        return arr;
    }

    // xt::arange: create a 1D array with values from start (inclusive) to stop (exclusive) in steps.
    template <typename T>
    xarray<T> arange(T start, T stop, T step = T(1)) {
        assert(step != T(0) && "Step must be non-zero");
        std::vector<T> values;
        if (step > T(0)) {
            for (T v = start; v < stop; v += step)
                values.push_back(v);
        }
        else {
            for (T v = start; v > stop; v += step)
                values.push_back(v);
        }
        xarray<T> arr({ values.size() });
        arr.data = values;
        return arr;
    }

    // xt::eye: create an identity matrix of size N x N.
    template <typename T>
    xarray<T> eye(size_t N) {
        xarray<T> arr({ N, N });
        std::fill(arr.data.begin(), arr.data.end(), T(0));
        // For a 2D array with shape {N, N}: strides[0] = N and strides[1] = 1.
        for (size_t i = 0; i < N; i++) {
            // Use the multi-index operator for clarity.
            arr({ i, i }) = T(1);
        }
        return arr;
    }

    // xt::reshape: reshape an array to a new shape (the total number of elements must match).
    template <typename T>
    xarray<T> reshape(const xarray<T>& arr, const std::vector<size_t>& new_shape) {
        size_t new_total = 1;
        for (auto s : new_shape) new_total *= s;
        assert(new_total == arr.size() && "Total size must remain the same for reshape");
        xarray<T> result(new_shape);
        result.data = arr.data;  // shallow copy of the data
        return result;
    }

    // xt::transpose: for a 2D array, swap rows and columns.
    template <typename T>
    xarray<T> transpose(const xarray<T>& arr) {
        assert(arr.shape.size() == 2 && "Transpose is implemented only for 2D arrays");
        std::vector<size_t> new_shape = { arr.shape[1], arr.shape[0] };
        xarray<T> result(new_shape);
        for (size_t i = 0; i < arr.shape[0]; i++) {
            for (size_t j = 0; j < arr.shape[1]; j++) {
                result({ j, i }) = arr({ i, j });
            }
        }
        return result;
    }

    // xt::stack: stack a vector of arrays along a new axis.
    // For simplicity, we only implement stacking along axis 0.
    template <typename T>
    xarray<T> stack(const std::vector<xarray<T>>& arrays, size_t axis = 0) {
        assert(!arrays.empty() && "Stacking requires at least one array");
        const auto& base_shape = arrays[0].shape;
        for (const auto& arr : arrays) {
            assert(arr.shape == base_shape && "All arrays must have the same shape to stack");
        }
        // Insert a new dimension at position 'axis'
        std::vector<size_t> new_shape = base_shape;
        new_shape.insert(new_shape.begin() + axis, arrays.size());
        xarray<T> result(new_shape);
        // For axis==0, each array's data is copied consecutively.
        size_t per_array = arrays[0].size();
        for (size_t i = 0; i < arrays.size(); i++) {
            std::copy(arrays[i].data.begin(),
                      arrays[i].data.end(),
                      result.data.begin() + i * per_array);
        }
        return result;
    }

    // xt::flip: flip an array along a given axis.
    template <typename T>
    xarray<T> flip(const xarray<T>& arr, size_t axis) {
        assert(axis < arr.shape.size() && "Axis out of bounds");
        xarray<T> result(arr.shape);
        // Helper: compute multi-index from a linear index.
        auto compute_multi_index = [&](size_t lin_idx) -> std::vector<size_t> {
            std::vector<size_t> idx(arr.shape.size());
            for (size_t i = 0; i < arr.shape.size(); i++) {
                idx[i] = (lin_idx / arr.strides[i]) % arr.shape[i];
            }
            return idx;
        };
        // Iterate over all elements.
        for (size_t lin = 0; lin < arr.size(); lin++) {
            auto idx = compute_multi_index(lin);
            // Flip the index along the specified axis.
            idx[axis] = arr.shape[axis] - 1 - idx[axis];
            // Compute the linear index in the result.
            size_t new_lin = 0;
            for (size_t i = 0; i < idx.size(); i++) {
                new_lin += idx[i] * result.strides[i];
            }
            result.data[new_lin] = arr.data[lin];
        }
        return result;
    }

} // namespace xt

//-------------------------------------
// Example usage of all functions.
//-------------------------------------
int main() {
    using namespace xt;

    // zeros & ones & empty
    auto z = zeros<double>({3, 4});
    auto o = ones<double>({2, 5});
    auto e = empty<double>({3, 3});

    std::cout << "zeros({3,4}) first element: " << z({0, 0}) << "\n";
    std::cout << "ones({2,5}) first element: " << o({0, 0}) << "\n";

    // linspace: 1D array from 0 to 1 with 5 points.
    auto ls = linspace<double>(0.0, 1.0, 5);
    std::cout << "linspace(0,1,5): ";
    for (auto val : ls.data)
        std::cout << val << " ";
    std::cout << "\n";

    // arange: 1D array from 0 to 10 (exclusive) with step 2.
    auto ar = arange<int>(0, 10, 2);
    std::cout << "arange(0,10,2): ";
    for (auto val : ar.data)
        std::cout << val << " ";
    std::cout << "\n";

    // eye: 4x4 identity matrix.
    auto I = eye<double>(4);
    std::cout << "eye(4):\n";
    for (size_t i = 0; i < I.shape[0]; i++) {
        for (size_t j = 0; j < I.shape[1]; j++) {
            std::cout << I({i, j}) << " ";
        }
        std::cout << "\n";
    }

    // reshape: reshape a 1D array of 12 elements into a 3x4 array.
    auto arr1d = arange<int>(0, 12, 1);
    auto reshaped = reshape(arr1d, {3, 4});
    std::cout << "Reshaped array (3x4):\n";
    for (size_t i = 0; i < reshaped.shape[0]; i++) {
        for (size_t j = 0; j < reshaped.shape[1]; j++) {
            std::cout << reshaped({i, j}) << "\t";
        }
        std::cout << "\n";
    }

    // transpose: transpose the 3x4 array to 4x3.
    auto transposed = transpose(reshaped);
    std::cout << "Transposed array (4x3):\n";
    for (size_t i = 0; i < transposed.shape[0]; i++) {
        for (size_t j = 0; j < transposed.shape[1]; j++) {
            std::cout << transposed({i, j}) << "\t";
        }
        std::cout << "\n";
    }

    // stack: stack two 2x2 arrays along a new axis (axis 0).
    auto a1 = ones<double>({2, 2});
    auto a2 = zeros<double>({2, 2});
    std::vector<xarray<double>> vec = { a1, a2 };
    auto stacked = stack(vec, 0); // new shape: {2,2,2}
    std::cout << "Stacked array shape: {";
    for (auto s : stacked.shape)
        std::cout << s << " ";
    std::cout << "}\n";

    // flip: flip the reshaped array along axis 0.
    auto flipped = flip(reshaped, 0);
    std::cout << "Flipped reshaped array (flip along axis 0):\n";
    for (size_t i = 0; i < flipped.shape[0]; i++) {
        for (size_t j = 0; j < flipped.shape[1]; j++) {
            std::cout << flipped({i, j}) << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}
