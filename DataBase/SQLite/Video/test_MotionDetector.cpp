/**
 * @file test_MotionDetector.cpp
 * @brief Unit tests for the MotionDetector class.
 *
 * Tests the database and utility APIs with defined inputs and expected
 * outputs. Video processing tests are excluded as they require actual
 * video files and GUI support.
 */

#include "MotionDetector.hpp"
#include <opencv2/opencv.hpp>
#include <cassert>
#include <iostream>
#include <cstdio>

/// @brief Temporary database file for tests.
static const std::string TEST_DB = "test_motion.db";

/**
 * @brief Remove the test database file.
 */
void CleanUp() {
    std::remove(TEST_DB.c_str());
}

/**
 * @brief Test database schema initialization.
 */
void TestInitSchema() {
    std::cout << "  TestInitSchema... ";
    CleanUp();

    MotionDetector detector(TEST_DB);
    assert(detector.InitSchema());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test inserting and retrieving motion events.
 */
void TestInsertAndGetEvents() {
    std::cout << "  TestInsertAndGetEvents... ";
    CleanUp();

    MotionDetector detector(TEST_DB);
    detector.InitSchema();

    assert(detector.InsertEvent("2025-05-17 01:30:00", "Motion detected"));
    assert(detector.InsertEvent("2025-05-17 02:00:00", "Motion detected"));

    auto events = detector.GetAllEvents();
    assert(events.size() == 2);
    assert(events[0].timestamp == "2025-05-17 01:30:00");
    assert(events[0].status == "Motion detected");
    assert(events[1].timestamp == "2025-05-17 02:00:00");

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test querying events by time range.
 */
void TestGetEventsByTimeRange() {
    std::cout << "  TestGetEventsByTimeRange... ";
    CleanUp();

    MotionDetector detector(TEST_DB);
    detector.InitSchema();

    detector.InsertEvent("2025-05-17 00:30:00");
    detector.InsertEvent("2025-05-17 01:30:00");
    detector.InsertEvent("2025-05-17 02:30:00");
    detector.InsertEvent("2025-05-17 03:30:00");

    auto events = detector.GetEventsByTimeRange(
        "2025-05-17 01:00:00", "2025-05-17 03:00:00");
    assert(events.size() == 2);
    assert(events[0].timestamp == "2025-05-17 01:30:00");
    assert(events[1].timestamp == "2025-05-17 02:30:00");

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test empty database returns no events.
 */
void TestEmptyDatabase() {
    std::cout << "  TestEmptyDatabase... ";
    CleanUp();

    MotionDetector detector(TEST_DB);
    detector.InitSchema();

    auto events = detector.GetAllEvents();
    assert(events.empty());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test MakeFilename sanitizes timestamps properly.
 */
void TestMakeFilename() {
    std::cout << "  TestMakeFilename... ";

    std::string filename = MotionDetector::MakeFilename(
        "2025-05-17 01:30:00", "events");

    // Spaces replaced with _, colons replaced with -
    assert(filename == "events/motion_2025-05-17_01-30-00.jpg");

    std::cout << "PASSED\n";
}

/**
 * @brief Test GetCurrentTimestamp returns valid format.
 */
void TestGetCurrentTimestamp() {
    std::cout << "  TestGetCurrentTimestamp... ";

    std::string ts = MotionDetector::GetCurrentTimestamp();
    assert(!ts.empty());
    assert(ts.size() == 19);
    assert(ts[4] == '-');
    assert(ts[10] == ' ');

    std::cout << "PASSED\n";
}

/**
 * @brief Test motion threshold getter.
 */
void TestMotionThreshold() {
    std::cout << "  TestMotionThreshold... ";
    CleanUp();

    MotionDetector defaultDetector(TEST_DB);
    assert(defaultDetector.GetMotionThreshold() == MotionDetector::DEFAULT_THRESHOLD);

    CleanUp();

    MotionDetector customDetector(TEST_DB, "events", 10000);
    assert(customDetector.GetMotionThreshold() == 10000);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test MotionEvent::ToString format.
 */
void TestMotionEventToString() {
    std::cout << "  TestMotionEventToString... ";

    MotionEvent event{42, "2025-05-17 01:30:00", "Motion detected"};
    std::string str = event.ToString();
    assert(str.find("id=42") != std::string::npos);
    assert(str.find("2025-05-17 01:30:00") != std::string::npos);
    assert(str.find("Motion detected") != std::string::npos);

    std::cout << "PASSED\n";
}

/**
 * @brief Test ComputeMotionLevel with identical frames (no motion).
 */
void TestComputeMotionLevelNoMotion() {
    std::cout << "  TestComputeMotionLevelNoMotion... ";

    cv::Mat gray = cv::Mat::zeros(100, 100, CV_8UC1);
    int level = MotionDetector::ComputeMotionLevel(gray, gray);
    assert(level == 0);

    std::cout << "PASSED\n";
}

/**
 * @brief Test ComputeMotionLevel with different frames (motion detected).
 */
void TestComputeMotionLevelWithMotion() {
    std::cout << "  TestComputeMotionLevelWithMotion... ";

    cv::Mat gray1 = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::Mat gray2 = cv::Mat::ones(100, 100, CV_8UC1) * 128;

    int level = MotionDetector::ComputeMotionLevel(gray1, gray2);
    assert(level > 0);

    std::cout << "PASSED\n";
}

/**
 * @brief Main test runner.
 */
int main() {
    std::cout << "Running MotionDetector tests...\n";

    TestInitSchema();
    TestInsertAndGetEvents();
    TestGetEventsByTimeRange();
    TestEmptyDatabase();
    TestMakeFilename();
    TestGetCurrentTimestamp();
    TestMotionThreshold();
    TestMotionEventToString();
    TestComputeMotionLevelNoMotion();
    TestComputeMotionLevelWithMotion();

    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
