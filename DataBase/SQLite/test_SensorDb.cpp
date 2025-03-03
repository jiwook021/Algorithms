/**
 * @file test_SensorDb.cpp
 * @brief Unit tests for the SensorDb class.
 *
 * Tests the public API with defined inputs and expected outputs.
 * Uses an in-memory SQLite database to avoid file system side effects.
 */

#include "SensorDb.hpp"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <cmath>

/// @brief Temporary database file for tests.
static const std::string TEST_DB = "test_sensor.db";

/**
 * @brief Remove the test database file.
 */
void CleanUp() {
    std::remove(TEST_DB.c_str());
}

/**
 * @brief Test that schema initialization succeeds.
 */
void TestInitSchema() {
    std::cout << "  TestInitSchema... ";
    CleanUp();

    SensorDb db(TEST_DB);
    assert(db.InitSchema());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test inserting a reading with a specific timestamp and retrieving it.
 */
void TestInsertAndGetAll() {
    std::cout << "  TestInsertAndGetAll... ";
    CleanUp();

    SensorDb db(TEST_DB);
    db.InitSchema();

    assert(db.InsertReading("2025-01-15 10:00:00", 22.5));
    assert(db.InsertReading("2025-01-15 11:00:00", 23.1));

    auto readings = db.GetAllReadings();
    assert(readings.size() == 2);
    assert(readings[0].timestamp == "2025-01-15 10:00:00");
    assert(std::fabs(readings[0].temperature - 22.5) < 0.01);
    assert(readings[1].timestamp == "2025-01-15 11:00:00");
    assert(std::fabs(readings[1].temperature - 23.1) < 0.01);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test querying readings by time range.
 */
void TestGetByTimeRange() {
    std::cout << "  TestGetByTimeRange... ";
    CleanUp();

    SensorDb db(TEST_DB);
    db.InitSchema();

    db.InsertReading("2025-01-15 08:00:00", 20.0);
    db.InsertReading("2025-01-15 10:00:00", 22.5);
    db.InsertReading("2025-01-15 12:00:00", 25.0);
    db.InsertReading("2025-01-15 14:00:00", 27.5);

    auto readings = db.GetReadingsByTimeRange(
        "2025-01-15 09:00:00", "2025-01-15 13:00:00");
    assert(readings.size() == 2);
    assert(std::fabs(readings[0].temperature - 22.5) < 0.01);
    assert(std::fabs(readings[1].temperature - 25.0) < 0.01);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test GetAllReadings on empty database returns empty vector.
 */
void TestEmptyDatabase() {
    std::cout << "  TestEmptyDatabase... ";
    CleanUp();

    SensorDb db(TEST_DB);
    db.InitSchema();

    auto readings = db.GetAllReadings();
    assert(readings.empty());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test GetReadingsByTimeRange with no matching records.
 */
void TestTimeRangeNoResults() {
    std::cout << "  TestTimeRangeNoResults... ";
    CleanUp();

    SensorDb db(TEST_DB);
    db.InitSchema();

    db.InsertReading("2025-01-15 10:00:00", 22.5);

    auto readings = db.GetReadingsByTimeRange(
        "2025-06-01 00:00:00", "2025-06-02 00:00:00");
    assert(readings.empty());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test SensorReading::ToString produces expected output.
 */
void TestSensorReadingToString() {
    std::cout << "  TestSensorReadingToString... ";

    SensorReading reading{1, "2025-01-15 10:00:00", 22.5};
    std::string str = reading.ToString();
    assert(str.find("id=1") != std::string::npos);
    assert(str.find("2025-01-15 10:00:00") != std::string::npos);
    assert(str.find("22.5") != std::string::npos);

    std::cout << "PASSED\n";
}

/**
 * @brief Test GetCurrentTimeString returns a non-empty string.
 */
void TestGetCurrentTimeString() {
    std::cout << "  TestGetCurrentTimeString... ";

    std::string ts = SensorDb::GetCurrentTimeString();
    assert(!ts.empty());
    // Should be in format YYYY-MM-DD HH:MM:SS (19 chars)
    assert(ts.size() == 19);
    assert(ts[4] == '-');
    assert(ts[10] == ' ');

    std::cout << "PASSED\n";
}

/**
 * @brief Test auto-increment IDs are assigned correctly.
 */
void TestAutoIncrementIds() {
    std::cout << "  TestAutoIncrementIds... ";
    CleanUp();

    SensorDb db(TEST_DB);
    db.InitSchema();

    db.InsertReading("2025-01-01 00:00:00", 10.0);
    db.InsertReading("2025-01-02 00:00:00", 20.0);
    db.InsertReading("2025-01-03 00:00:00", 30.0);

    auto readings = db.GetAllReadings();
    assert(readings.size() == 3);
    assert(readings[0].id == 1);
    assert(readings[1].id == 2);
    assert(readings[2].id == 3);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Main test runner.
 */
int main() {
    std::cout << "Running SensorDb tests...\n";

    TestInitSchema();
    TestInsertAndGetAll();
    TestGetByTimeRange();
    TestEmptyDatabase();
    TestTimeRangeNoResults();
    TestSensorReadingToString();
    TestGetCurrentTimeString();
    TestAutoIncrementIds();

    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
