/**
 * @file SensorDb.hpp
 * @brief SQLite-based sensor data logger for storing and querying sensor readings.
 *
 * Wraps SQLite3 C API in a class-based interface for inserting timestamped
 * sensor data and retrieving logged records.
 */

#ifndef SENSOR_DB_HPP
#define SENSOR_DB_HPP

#include <string>
#include <vector>
#include <memory>

// Forward declare sqlite3 to avoid exposing the header
struct sqlite3;

/**
 * @struct SensorReading
 * @brief Represents a single sensor data point.
 */
struct SensorReading {
    int id;                 ///< Auto-increment row ID.
    std::string timestamp;  ///< ISO-8601 timestamp string.
    double temperature;     ///< Temperature value.

    /**
     * @brief Format reading as a human-readable string.
     * @return Formatted string.
     */
    std::string ToString() const;
};

/**
 * @class SensorDb
 * @brief Manages a SQLite database for sensor data logging.
 *
 * Provides methods to initialize the schema, insert readings,
 * query all readings, and query by time range.
 */
class SensorDb {
public:
    /**
     * @brief Construct and open (or create) the database.
     * @param dbPath Path to the SQLite database file.
     * @throws std::runtime_error if the database cannot be opened.
     */
    explicit SensorDb(const std::string& dbPath);

    /**
     * @brief Close the database connection.
     */
    ~SensorDb();

    /// @brief Deleted copy operations (non-copyable resource).
    SensorDb(const SensorDb&) = delete;
    SensorDb& operator=(const SensorDb&) = delete;

    /**
     * @brief Create the SensorLog table if it does not exist.
     * @return True on success.
     */
    bool InitSchema();

    /**
     * @brief Insert a sensor reading with the current timestamp.
     * @param temperature Temperature value to log.
     * @return True on success.
     */
    bool InsertReading(double temperature);

    /**
     * @brief Insert a sensor reading with a specific timestamp.
     * @param timestamp ISO-8601 timestamp string.
     * @param temperature Temperature value to log.
     * @return True on success.
     */
    bool InsertReading(const std::string& timestamp, double temperature);

    /**
     * @brief Retrieve all sensor readings.
     * @return Vector of SensorReading structs.
     */
    std::vector<SensorReading> GetAllReadings();

    /**
     * @brief Retrieve sensor readings within a time range.
     * @param startTime Start of range (inclusive).
     * @param endTime End of range (inclusive).
     * @return Vector of SensorReading structs.
     */
    std::vector<SensorReading> GetReadingsByTimeRange(
        const std::string& startTime, const std::string& endTime);

    /**
     * @brief Get the current time as an ISO-8601 string.
     * @return Timestamp string (YYYY-MM-DD HH:MM:SS).
     */
    static std::string GetCurrentTimeString();

    /**
     * @brief Get the database file path.
     * @return Database path string.
     */
    const std::string& GetDbPath() const;

private:
    std::string dbPath_;  ///< Path to the database file.
    sqlite3* db_;         ///< SQLite database handle.

    /**
     * @brief Execute a simple SQL statement (no results).
     * @param sql SQL string to execute.
     * @return True on success.
     */
    bool ExecuteSql(const std::string& sql);
};

#endif // SENSOR_DB_HPP
