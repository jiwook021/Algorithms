/**
 * @file MotionDetector.cpp
 * @brief Implementation of the MotionDetector class.
 *
 * Uses OpenCV for frame processing and SQLite3 for event persistence.
 */

#include "MotionDetector.hpp"
#include <opencv2/opencv.hpp>
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <filesystem>

// ============================================================================
// MotionEvent
// ============================================================================

std::string MotionEvent::ToString() const {
    std::ostringstream oss;
    oss << "id=" << id
        << " timestamp=" << timestamp
        << " status=" << status;
    return oss.str();
}

// ============================================================================
// MotionDetector
// ============================================================================

MotionDetector::MotionDetector(const std::string& dbPath,
                               const std::string& eventsDir,
                               int motionThreshold)
    : dbPath_(dbPath),
      eventsDir_(eventsDir),
      motionThreshold_(motionThreshold),
      db_(nullptr) {
    if (sqlite3_open(dbPath.c_str(), &db_) != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db_) << "\n";
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

MotionDetector::~MotionDetector() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool MotionDetector::InitSchema() {
    if (!db_) return false;

    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS MotionEvents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL
        );
    )";
    char* errMsg = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << "\n";
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}

bool MotionDetector::InsertEvent(const std::string& timestamp,
                                  const std::string& status) {
    if (!db_) return false;

    const std::string sql =
        "INSERT INTO MotionEvents (timestamp, status) VALUES (?, ?);";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db_) << "\n";
        return false;
    }

    sqlite3_bind_text(stmt, 1, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, status.c_str(), -1, SQLITE_TRANSIENT);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return success;
}

int MotionDetector::ComputeMotionLevel(const cv::Mat& grayPrev,
                                        const cv::Mat& grayCurr) {
    cv::Mat delta, thresh;
    cv::absdiff(grayPrev, grayCurr, delta);
    cv::threshold(delta, thresh, 25, 255, cv::THRESH_BINARY);
    cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);
    return cv::countNonZero(thresh);
}

std::string MotionDetector::MakeFilename(const std::string& timestamp,
                                          const std::string& directory) {
    std::string filename = directory + "/motion_" + timestamp + ".jpg";
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '-');
    return filename;
}

std::string MotionDetector::GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t nowC = std::chrono::system_clock::to_time_t(now);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&nowC));
    return std::string(buffer);
}

int MotionDetector::GetMotionThreshold() const {
    return motionThreshold_;
}

int MotionDetector::ProcessVideo(const std::string& videoPath, bool showGui) {
    std::filesystem::create_directory(eventsDir_);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file: " << videoPath << "\n";
        return -1;
    }

    cv::Mat prevFrame, currFrame, grayPrev, grayCurr;

    cap >> prevFrame;
    if (prevFrame.empty()) {
        std::cerr << "Empty first frame.\n";
        return -1;
    }

    cv::cvtColor(prevFrame, grayPrev, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayPrev, grayPrev, cv::Size(21, 21), 0);

    int eventCount = 0;

    while (true) {
        cap >> currFrame;
        if (currFrame.empty()) break;

        cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayCurr, grayCurr, cv::Size(21, 21), 0);

        int motionLevel = ComputeMotionLevel(grayPrev, grayCurr);
        if (motionLevel > motionThreshold_) {
            std::string timestamp = GetCurrentTimestamp();
            InsertEvent(timestamp);

            std::string filename = MakeFilename(timestamp, eventsDir_);
            cv::imwrite(filename, currFrame);

            ++eventCount;
        }

        if (showGui) {
            cv::imshow("Video Playback", currFrame);
            if (cv::waitKey(30) == 'q') break;
        }

        grayPrev = grayCurr.clone();
    }

    cap.release();
    if (showGui) {
        cv::destroyAllWindows();
    }

    return eventCount;
}

std::vector<MotionEvent> MotionDetector::GetAllEvents() {
    std::vector<MotionEvent> events;
    if (!db_) return events;

    const std::string sql =
        "SELECT id, timestamp, status FROM MotionEvents ORDER BY id;";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return events;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MotionEvent event;
        event.id = sqlite3_column_int(stmt, 0);
        const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        event.timestamp = ts ? ts : "";
        const char* st = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        event.status = st ? st : "";
        events.push_back(event);
    }

    sqlite3_finalize(stmt);
    return events;
}

std::vector<MotionEvent> MotionDetector::GetEventsByTimeRange(
    const std::string& startTime, const std::string& endTime) {

    std::vector<MotionEvent> events;
    if (!db_) return events;

    const std::string sql =
        "SELECT id, timestamp, status FROM MotionEvents "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC;";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return events;
    }

    sqlite3_bind_text(stmt, 1, startTime.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, endTime.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MotionEvent event;
        event.id = sqlite3_column_int(stmt, 0);
        const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        event.timestamp = ts ? ts : "";
        const char* st = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        event.status = st ? st : "";
        events.push_back(event);
    }

    sqlite3_finalize(stmt);
    return events;
}
