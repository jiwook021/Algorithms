/**
 * @file Bitset.hpp
 * @brief Custom bitset implementation with packed bit storage
 * @details Wraps a uint64_t to represent a fixed-size bit sequence (up to 64 bits).
 *          Supports Set, Reset, Flip, Test, and count with O(1) random access.
 */

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <limits>

template <std::size_t N>
class MyBitset {
    static_assert(N > 0 && N <= 64, "This implementation supports only up to 64 bits.");

private:
    uint64_t bits_ = 0;

    static constexpr uint64_t FullMask() {
        return (N == 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << N) - 1);
    }

public:
    void Set(std::size_t pos) {
        if (pos >= N) throw std::out_of_range("Index out of range");
        bits_ |= (1ULL << pos);
    }

    void Reset(std::size_t pos) {
        if (pos >= N) throw std::out_of_range("Index out of range");
        bits_ &= ~(1ULL << pos);
    }

    void Flip(std::size_t pos) {
        if (pos >= N) throw std::out_of_range("Index out of range");
        bits_ ^= (1ULL << pos);
    }

    void SetAll() { bits_ = FullMask(); }

    void ResetAll() { bits_ = 0; }

    void FlipAll() { bits_ ^= FullMask(); }

    bool Test(std::size_t pos) const {
        if (pos >= N) throw std::out_of_range("Index out of range");
        return bits_ & (1ULL << pos);
    }

    bool All() const { return bits_ == FullMask(); }

    bool Any() const { return bits_ != 0; }

    bool None() const { return bits_ == 0; }

    std::string ToString() const {
        std::string s;
        s.reserve(N);
        for (std::size_t i = N; i > 0; --i) {
            s += ((bits_ >> (i - 1)) & 1) ? '1' : '0';
        }
        return s;
    }

    void Print() const {
        std::cout << ToString() << '\n';
    }
};
