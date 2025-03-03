/**
 * @file VideoUploader.hpp
 * @brief HTTP video file uploader using libcurl
 * @details Uploads a video file to a remote server via HTTP multipart form
 *          with progress reporting.
 */

#pragma once

#include <string>
#include <functional>

namespace VideoUpload {

/// Progress callback: (uploadedBytes, totalBytes).
using ProgressCallback = std::function<void(long long, long long)>;

/// Result of an upload operation.
struct UploadResult {
    bool success;
    long httpCode;
    std::string responseBody;
    std::string errorMessage;
};

/// Extract the filename component from a file path.
std::string GetFilename(const std::string& path);

/**
 * Upload a video file to a server via HTTP multipart POST.
 *
 * @param filePath   Path to the video file.
 * @param serverUrl  URL of the upload endpoint.
 * @param onProgress Optional progress callback.
 * @return UploadResult with HTTP code and response body.
 */
UploadResult UploadFile(const std::string& filePath,
                        const std::string& serverUrl,
                        ProgressCallback onProgress = nullptr);

} // namespace VideoUpload
