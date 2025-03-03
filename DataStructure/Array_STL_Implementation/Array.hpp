// Array.hpp
// A fixed-size container that mimics std::array behavior.
// Stores N elements of type T in contiguous stack memory.
// Supports random access, bounds-checked access, forward/reverse iteration,
// and is compatible with STL algorithms (std::sort, std::find, etc.).
//
// This implementation intentionally uses raw types (T*, T&, std::size_t, etc.)
// directly instead of type aliases, so every signature is explicit and readable.

#pragma once

#include <cstddef>       // std::size_t, std::ptrdiff_t
#include <stdexcept>     // std::out_of_range
#include <iterator>      // std::reverse_iterator
#include <algorithm>     // std::fill_n, std::swap_ranges
#include <type_traits>   // std::is_nothrow_swappable_v

template<typename T, std::size_t N>
class Array {
public:

    // The underlying fixed-size C array that holds all elements.
    // Public so that Array remains an aggregate (supports brace initialization).
    T MData[N];

    // ---- Element Access ----

    // Returns element at 'index' without bounds checking (undefined behavior if out of range).
    constexpr T& operator[](std::size_t Index) noexcept             { return MData[Index]; }
    constexpr const T& operator[](std::size_t Index) const noexcept { return MData[Index]; }

    // Returns element at 'index' with bounds checking; throws std::out_of_range if invalid.
    constexpr T& At(std::size_t Index) {
        if (Index >= N) throw std::out_of_range("Array::at(): index out of range");
        return MData[Index];
    }
    constexpr const T& At(std::size_t Index) const {
        if (Index >= N) throw std::out_of_range("Array::at(): index out of range");
        return MData[Index];
    }

    // Returns a reference to the first element.
    constexpr T& Front() noexcept             { return MData[0]; }
    constexpr const T& Front() const noexcept { return MData[0]; }

    // Returns a reference to the last element.
    constexpr T& Back() noexcept              { return MData[N - 1]; }
    constexpr const T& Back() const noexcept  { return MData[N - 1]; }

    // Returns a raw pointer to the underlying contiguous storage.
    constexpr T* data() noexcept              { return MData; }
    constexpr const T* data() const noexcept  { return MData; }

    // ---- Forward Iterators ----

    // Points to the first element.
    constexpr T* begin() noexcept              { return MData; }
    constexpr const T* begin() const noexcept  { return MData; }
    constexpr const T* cbegin() const noexcept { return MData; }

    // Points one past the last element.
    constexpr T* end() noexcept                { return MData + N; }
    constexpr const T* end() const noexcept    { return MData + N; }
    constexpr const T* cend() const noexcept   { return MData + N; }

    // ---- Reverse Iterators ----

    // Points to the last element (iterates backward toward the first).
    constexpr std::reverse_iterator<T*> rbegin() noexcept              { return std::reverse_iterator<T*>(end()); }
    constexpr std::reverse_iterator<const T*> rbegin() const noexcept  { return std::reverse_iterator<const T*>(end()); }
    constexpr std::reverse_iterator<const T*> crbegin() const noexcept { return std::reverse_iterator<const T*>(cend()); }

    // Points one before the first element (reverse-end sentinel).
    constexpr std::reverse_iterator<T*> rend() noexcept                { return std::reverse_iterator<T*>(begin()); }
    constexpr std::reverse_iterator<const T*> rend() const noexcept    { return std::reverse_iterator<const T*>(begin()); }
    constexpr std::reverse_iterator<const T*> crend() const noexcept   { return std::reverse_iterator<const T*>(cbegin()); }

    // ---- Capacity ----

    // Returns the number of elements (always N, the compile-time size).
    [[nodiscard]] constexpr std::size_t Size() const noexcept     { return N; }

    // Returns the maximum possible number of elements (same as size() for a fixed-size array).
    [[nodiscard]] constexpr std::size_t max_size() const noexcept { return N; }

    // Returns true if the array has zero elements (N == 0).
    [[nodiscard]] constexpr bool Empty() const noexcept           { return N == 0; }

    // ---- Operations ----

    // Sets every element in the array to 'value'.
    constexpr void fill(const T& value) {
        std::fill_n(begin(), N, value);
    }

    // Swaps all elements with another Array of the same type and size.
    constexpr void Swap(Array& Other) noexcept(std::is_nothrow_swappable_v<T>) {
        std::swap_ranges(begin(), end(), Other.begin());
    }
};

// Free-function swap so that ADL (Argument-Dependent Lookup) finds it
// when calling swap(a, b) without the std:: prefix.
template<typename T, std::size_t N>
constexpr void Swap(Array<T, N>& a, Array<T, N>& b) noexcept(noexcept(a.Swap(b))) {
    a.Swap(b);
}
