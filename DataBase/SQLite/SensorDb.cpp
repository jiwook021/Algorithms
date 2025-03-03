/**
 * @file SensorDb.cpp
 * @brief Implementation of the SensorDb class.
 *
 * Uses the SQLite3 C API to manage a sensor data log database.
 */

#include "SensorDb.hpp"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <ctime>
#include <stdexcept>

// ============================================================================
// SensorReading
// ============================================================================

std::string SensorReading::ToString() const {
    std::ostringstream oss;
    oss << "id=" << id
        << " timestamp=" << timestamp
        << " temperature=" << temperature;
    return oss.str();
}

// ============================================================================
// SensorDb
// ============================================================================

SensorDb::SensorDb(const std::string& dbPath)
    : dbPath_(dbPath), db_(nullptr) {
    if (sqlite3_open(dbPath.c_str(), &db_) != SQLITE_OK) {
        std::string err = "Failed to open database: ";
        err += sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error(err);
    }
}

SensorDb::~SensorDb() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool SensorDb::InitSchema() {
    const std::string sql = R"(
        CREATE TABLE IF NOT EXISTS SensorLog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL
        );
    )";
    return ExecuteSql(sql);
}

bool SensorDb::InsertReading(double temperature) {
    return InsertReading(GetCurrentTimeString(), temperature);
}

bool SensorDb::InsertReading(const std::string& timestamp, double temperature) {
    const std::string sql =
        "INSERT INTO SensorLog (timestamp, temperature) VALUES (?, ?);";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "SQL prepare error: " << sqlite3_errmsg(db_) << "\n";
        return false;
    }

    sqlite3_bind_text(stmt, 1, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 2, temperature);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    if (!success) {
        std::cerr << "SQL step error: " << sqlite3_errmsg(db_) << "\n";
    }

    sqlite3_finalize(stmt);
    return success;
}

std::vector<SensorReading> SensorDb::GetAllReadings() {
    std::vector<SensorReading> readings;
    const std::string sql = "SELECT id, timestamp, temperature FROM SensorLog ORDER BY id;";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "SQL prepare error: " << sqlite3_errmsg(db_) << "\n";
        return readings;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SensorReading reading;
        reading.id = sqlite3_column_int(stmt, 0);
        const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        reading.timestamp = ts ? ts : "";
        reading.temperature = sqlite3_column_double(stmt, 2);
        readings.push_back(reading);
    }

    sqlite3_finalize(stmt);
    return readings;
}

std::vector<SensorReading> SensorDb::GetReadingsByTimeRange(
    const std::string& startTime, const std::string& endTime) {

    std::vector<SensorReading> readings;
    const std::string sql =
        "SELECT id, timestamp, temperature FROM SensorLog "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC;";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "SQL prepare error: " << sqlite3_errmsg(db_) << "\n";
        return readings;
    }

    sqlite3_bind_text(stmt, 1, startTime.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, endTime.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SensorReading reading;
        reading.id = sqlite3_column_int(stmt, 0);
        const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        reading.timestamp = ts ? ts : "";
        reading.temperature = sqlite3_column_double(stmt, 2);
        readings.push_back(reading);
    }

    sqlite3_finalize(stmt);
    return readings;
}

std::string SensorDb::GetCurrentTimeString() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

const std::string& SensorDb::GetDbPath() const {
    return dbPath_;
}

bool SensorDb::ExecuteSql(const std::string& sql) {
    char* errMsg = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << "\n";
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}
