/**
 * @file main.cpp
 * @brief Driver for the VideoUploader module.
 */

#include "VideoUploader.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file> <server_url>\n";
        std::cerr << "Example: " << argv[0]
                  << " video.mp4 http://192.168.1.100:8080/upload\n";
        return 1;
    }

    const std::string videoPath = argv[1];
    const std::string serverUrl = argv[2];

    std::cout << "Uploading: " << VideoUpload::GetFilename(videoPath) << "\n";
    std::cout << "To: " << serverUrl << "\n";

    auto result = VideoUpload::UploadFile(
        videoPath, serverUrl,
        [](long long now, long long total) {
            double pct = 100.0 * static_cast<double>(now) /
                         static_cast<double>(total);
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                      << pct << "%" << std::flush;
        });

    std::cout << "\n";

    if (result.success) {
        std::cout << "Upload successful (HTTP " << result.httpCode << ")\n";
        std::cout << "Response: " << result.responseBody << "\n";
    } else {
        std::cerr << "Upload failed: " << result.errorMessage << "\n";
        return 1;
    }

    return 0;
}
