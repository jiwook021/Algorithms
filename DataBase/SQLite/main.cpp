/**
 * @file main.cpp
 * @brief Entry point for the SensorDb module.
 *
 * Demonstrates inserting simulated sensor readings and querying them back.
 */

#include "SensorDb.hpp"
#include <iostream>

/**
 * @brief Main entry point. Creates a sensor database, inserts sample data,
 *        and prints all logged readings.
 * @return 0 on success, 1 on failure.
 */
int main() {
    try {
        SensorDb db("sensor_data.db");

        if (!db.InitSchema()) {
            std::cerr << "Failed to initialize schema.\n";
            return 1;
        }

        // Insert a simulated sensor reading
        double simulatedTemperature = 25.3;
        if (!db.InsertReading(simulatedTemperature)) {
            std::cerr << "Failed to insert reading.\n";
            return 1;
        }
        std::cout << "Inserted temperature: " << simulatedTemperature << "\n";

        // Retrieve and display all readings
        auto readings = db.GetAllReadings();
        std::cout << "\nLogged Data (" << readings.size() << " records):\n";
        for (const auto& reading : readings) {
            std::cout << "  " << reading.ToString() << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
