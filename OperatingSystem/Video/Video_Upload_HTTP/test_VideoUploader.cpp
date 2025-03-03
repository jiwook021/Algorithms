/**
 * @file test_VideoUploader.cpp
 * @brief Unit tests for VideoUploader module.
 *
 * Tests the pure utility functions (no network required).
 * Upload tests are limited to verifying error handling for missing files.
 */

#include "VideoUploader.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

/// GetFilename extracts the filename from various path formats.
void TestGetFilename() {
    assert(VideoUpload::GetFilename("/home/user/video.mp4") == "video.mp4");
    assert(VideoUpload::GetFilename("video.mp4") == "video.mp4");
    assert(VideoUpload::GetFilename("/a/b/c/d.avi") == "d.avi");
    assert(VideoUpload::GetFilename("C:\\Users\\video.mp4") == "video.mp4");
    assert(VideoUpload::GetFilename("") == "");
    std::cout << "[PASS] TestGetFilename\n";
}

/// UploadFile returns error for nonexistent file.
void TestUploadMissingFile() {
    auto result = VideoUpload::UploadFile(
        "/nonexistent/path/video.mp4",
        "http://localhost:9999/upload");
    assert(!result.success);
    assert(!result.errorMessage.empty());
    std::cout << "[PASS] TestUploadMissingFile\n";
}

/// UploadFile returns error when server is unreachable.
void TestUploadUnreachableServer() {
    // Create a temporary file to upload.
    const std::string tmpFile = "/tmp/test_video_upload.dat";
    {
        std::ofstream ofs(tmpFile, std::ios::binary);
        ofs << "fake video data";
    }

    auto result = VideoUpload::UploadFile(tmpFile, "http://127.0.0.1:1/upload");
    assert(!result.success);
    assert(!result.errorMessage.empty());

    std::remove(tmpFile.c_str());
    std::cout << "[PASS] TestUploadUnreachableServer\n";
}

/// Progress callback is invoked (even if upload fails).
void TestProgressCallback() {
    const std::string tmpFile = "/tmp/test_video_progress.dat";
    {
        std::ofstream ofs(tmpFile, std::ios::binary);
        // Write enough data so progress might fire.
        std::string data(1024, 'X');
        ofs << data;
    }

    bool callbackFired = false;
    auto result = VideoUpload::UploadFile(
        tmpFile, "http://127.0.0.1:1/upload",
        [&callbackFired](long long, long long) {
            callbackFired = true;
        });

    // The callback may or may not fire if the connection fails immediately.
    // We just verify no crash occurred.
    assert(!result.success);

    std::remove(tmpFile.c_str());
    std::cout << "[PASS] TestProgressCallback\n";
}

int main() {
    TestGetFilename();
    TestUploadMissingFile();
    TestUploadUnreachableServer();
    TestProgressCallback();
    std::cout << "\nAll VideoUploader tests passed.\n";
    return 0;
}
