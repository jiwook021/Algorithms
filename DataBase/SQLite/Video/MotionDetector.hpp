/**
 * @file MotionDetector.hpp
 * @brief Video-based motion detection with SQLite event logging.
 *
 * Provides a MotionDetector class that processes video frames to detect
 * motion using frame differencing, logs events to a SQLite database,
 * and saves captured frames as images.
 *
 * Also provides a MotionViewer class for replaying logged motion events.
 */

#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <string>
#include <vector>

// Forward declarations to avoid exposing OpenCV/SQLite headers
struct sqlite3;

namespace cv {
    class Mat;
    class VideoCapture;
}

/**
 * @struct MotionEvent
 * @brief Represents a single motion detection event.
 */
struct MotionEvent {
    int id;                 ///< Database row ID.
    std::string timestamp;  ///< ISO-8601 timestamp.
    std::string status;     ///< Status description.

    /**
     * @brief Format event as a human-readable string.
     * @return Formatted string.
     */
    std::string ToString() const;
};

/**
 * @class MotionDetector
 * @brief Detects motion in video frames and logs events to SQLite.
 *
 * Uses Gaussian blur + frame differencing + thresholding to detect
 * motion. When motion exceeds a threshold, the event is logged to
 * the database and the frame is saved as a JPEG image.
 */
class MotionDetector {
public:
    /// @brief Default motion threshold (pixel count).
    static constexpr int DEFAULT_THRESHOLD = 5000;

    /**
     * @brief Construct a MotionDetector.
     * @param dbPath Path to the SQLite database file.
     * @param eventsDir Directory for saving captured frame images.
     * @param motionThreshold Minimum non-zero pixel count to trigger detection.
     */
    explicit MotionDetector(const std::string& dbPath,
                            const std::string& eventsDir = "events",
                            int motionThreshold = DEFAULT_THRESHOLD);

    /**
     * @brief Destructor. Closes database connection.
     */
    ~MotionDetector();

    /// @brief Non-copyable.
    MotionDetector(const MotionDetector&) = delete;
    MotionDetector& operator=(const MotionDetector&) = delete;

    /**
     * @brief Initialize the database schema.
     * @return True on success.
     */
    bool InitSchema();

    /**
     * @brief Insert a motion event into the database.
     * @param timestamp ISO-8601 timestamp.
     * @param status Event status string.
     * @return True on success.
     */
    bool InsertEvent(const std::string& timestamp,
                     const std::string& status = "Motion detected");

    /**
     * @brief Compute the motion level between two grayscale frames.
     * @param grayPrev Previous grayscale frame (blurred).
     * @param grayCurr Current grayscale frame (blurred).
     * @return Number of non-zero pixels in the thresholded difference.
     */
    static int ComputeMotionLevel(const cv::Mat& grayPrev,
                                  const cv::Mat& grayCurr);

    /**
     * @brief Generate a sanitized filename from a timestamp.
     * @param timestamp ISO-8601 timestamp.
     * @param directory Output directory.
     * @return Full path to the output file.
     */
    static std::string MakeFilename(const std::string& timestamp,
                                    const std::string& directory = "events");

    /**
     * @brief Get the current time as an ISO-8601 string.
     * @return Timestamp string.
     */
    static std::string GetCurrentTimestamp();

    /**
     * @brief Get the motion threshold.
     * @return Threshold value.
     */
    int GetMotionThreshold() const;

    /**
     * @brief Process a video file for motion detection.
     * @param videoPath Path to the video file.
     * @param showGui If true, display frames in a window.
     * @return Number of motion events detected.
     */
    int ProcessVideo(const std::string& videoPath, bool showGui = false);

    /**
     * @brief Query all motion events from the database.
     * @return Vector of MotionEvent structs.
     */
    std::vector<MotionEvent> GetAllEvents();

    /**
     * @brief Query motion events within a time range.
     * @param startTime Start of range (inclusive).
     * @param endTime End of range (inclusive).
     * @return Vector of MotionEvent structs.
     */
    std::vector<MotionEvent> GetEventsByTimeRange(
        const std::string& startTime, const std::string& endTime);

private:
    std::string dbPath_;          ///< Path to database file.
    std::string eventsDir_;       ///< Directory for frame images.
    int motionThreshold_;         ///< Motion pixel threshold.
    sqlite3* db_;                 ///< SQLite database handle.
};

#endif // MOTION_DETECTOR_HPP
