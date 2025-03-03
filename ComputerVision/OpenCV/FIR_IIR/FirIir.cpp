/**
 * @file FirIir.cpp
 * @brief Implementation of FIR and IIR filters
 */

#include "FirIir.hpp"

namespace FirIir {

FirFilter::FirFilter(const std::vector<double>& taps) : taps_(taps), writeIndex_(0) {
    if (taps_.empty()) throw std::invalid_argument("FIR taps must not be empty");
    buffer_.assign(taps_.size(), 0.0);
}

double FirFilter::Process(double input) {
    buffer_[writeIndex_] = input;
    double output = 0.0;
    size_t idx = writeIndex_;
    for (size_t k = 0; k < taps_.size(); ++k) {
        output += taps_[k] * buffer_[idx];
        idx = (idx == 0) ? buffer_.size() - 1 : idx - 1;
    }
    writeIndex_ = (writeIndex_ + 1) % buffer_.size();
    return output;
}

IirFilter::IirFilter(double alpha) : alpha_(alpha), prevOutput_(0.0), initialized_(false) {
    if (!(alpha > 0.0 && alpha < 1.0))
        throw std::invalid_argument("IIR alpha must be in (0,1)");
}

double IirFilter::Process(double input) {
    if (!initialized_) { prevOutput_ = input; initialized_ = true; }
    double output = alpha_ * input + (1.0 - alpha_) * prevOutput_;
    prevOutput_ = output;
    return output;
}

}  // namespace FirIir
