/**
 * @file test_Temperature.cpp
 * @brief Unit tests for Temperature conversion
 */

#include "Temperature.hpp"
#include <gtest/gtest.h>

using namespace Temperature;

TEST(Temperature, CelsiusToFahrenheitBoiling) {
    EXPECT_NEAR(CelsiusToFahrenheit(100.0), 212.0, 0.01);
}

TEST(Temperature, CelsiusToFahrenheitFreezing) {
    EXPECT_NEAR(CelsiusToFahrenheit(0.0), 32.0, 0.01);
}

TEST(Temperature, CelsiusToFahrenheitBodyTemp) {
    EXPECT_NEAR(CelsiusToFahrenheit(37.0), 98.6, 0.01);
}

TEST(Temperature, CelsiusToFahrenheitNegative) {
    EXPECT_NEAR(CelsiusToFahrenheit(-40.0), -40.0, 0.01);
}

TEST(Temperature, FahrenheitToCelsiusBoiling) {
    EXPECT_NEAR(FahrenheitToCelsius(212.0), 100.0, 0.01);
}

TEST(Temperature, FahrenheitToCelsiusFreezing) {
    EXPECT_NEAR(FahrenheitToCelsius(32.0), 0.0, 0.01);
}

TEST(Temperature, FahrenheitToCelsiusNegative) {
    EXPECT_NEAR(FahrenheitToCelsius(-40.0), -40.0, 0.01);
}

TEST(Temperature, RoundTripCelsius) {
    double original = 37.0;
    double roundTrip = FahrenheitToCelsius(CelsiusToFahrenheit(original));
    EXPECT_NEAR(roundTrip, original, 0.01);
}

TEST(Temperature, RoundTripFahrenheit) {
    double original = 98.6;
    double roundTrip = CelsiusToFahrenheit(FahrenheitToCelsius(original));
    EXPECT_NEAR(roundTrip, original, 0.01);
}
