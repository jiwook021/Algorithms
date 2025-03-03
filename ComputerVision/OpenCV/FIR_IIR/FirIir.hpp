/**
 * @file FirIir.hpp
 * @brief FIR and IIR digital filter implementations
 * @details FIR uses circular buffer convolution; IIR uses first-order
 *          exponential smoothing y[n] = a*x[n] + (1-a)*y[n-1].
 */

#pragma once

#include <vector>
#include <stdexcept>

namespace FirIir {

class FirFilter {
public:
    explicit FirFilter(const std::vector<double>& taps);
    double Process(double input);
private:
    std::vector<double> taps_;
    std::vector<double> buffer_;
    size_t writeIndex_;
};

class IirFilter {
public:
    explicit IirFilter(double alpha);
    double Process(double input);
private:
    double alpha_;
    double prevOutput_;
    bool initialized_;
};

}  // namespace FirIir
