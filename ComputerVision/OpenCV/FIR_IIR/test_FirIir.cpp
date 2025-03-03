/**
 * @file test_FirIir.cpp
 * @brief Unit tests for FirIir module
 */

#include "FirIir.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

namespace {

void TestFirEmptyTapsThrows() {
    bool threw = false;
    try { FirIir::FirFilter fir({}); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestFirEmptyTapsThrows PASSED\n";
}

void TestFirIdentityTap() {
    FirIir::FirFilter fir({1.0});
    assert(fir.Process(5.0) == 5.0);
    assert(fir.Process(3.0) == 3.0);
    std::cout << "  TestFirIdentityTap PASSED\n";
}

void TestFirAverageTaps() {
    FirIir::FirFilter fir({0.5, 0.5});
    // First sample: 0.5*10 + 0.5*0 = 5.0
    assert(std::abs(fir.Process(10.0) - 5.0) < 1e-10);
    // Second sample: 0.5*20 + 0.5*10 = 15.0
    assert(std::abs(fir.Process(20.0) - 15.0) < 1e-10);
    std::cout << "  TestFirAverageTaps PASSED\n";
}

void TestIirInvalidAlphaThrows() {
    bool threw = false;
    try { FirIir::IirFilter iir(0.0); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { FirIir::IirFilter iir(1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestIirInvalidAlphaThrows PASSED\n";
}

void TestIirFirstSample() {
    FirIir::IirFilter iir(0.5);
    // First sample initialises to input
    double out = iir.Process(10.0);
    assert(std::abs(out - 10.0) < 1e-10);
    std::cout << "  TestIirFirstSample PASSED\n";
}

void TestIirSmoothing() {
    FirIir::IirFilter iir(0.1);
    iir.Process(0.0);  // init
    double out = iir.Process(100.0);
    // 0.1*100 + 0.9*0 = 10
    assert(std::abs(out - 10.0) < 1e-10);
    std::cout << "  TestIirSmoothing PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running FirIir tests...\n";
    TestFirEmptyTapsThrows();
    TestFirIdentityTap();
    TestFirAverageTaps();
    TestIirInvalidAlphaThrows();
    TestIirFirstSample();
    TestIirSmoothing();
    std::cout << "All FirIir tests PASSED.\n";
    return 0;
}
