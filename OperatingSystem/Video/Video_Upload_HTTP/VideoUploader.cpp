/**
 * @file VideoUploader.cpp
 * @brief Implementation of the HTTP video uploader.
 */

#include "VideoUploader.hpp"
#include <curl/curl.h>
#include <fstream>
#include <iostream>

namespace VideoUpload {

std::string GetFilename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

// Internal: libcurl write callback.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                             void* userp) {
    size_t totalSize = size * nmemb;
    auto* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Internal: libcurl progress callback.
struct ProgressContext {
    ProgressCallback callback;
};

static int CurlProgressCallback(void* clientp, curl_off_t /*dlTotal*/,
                                  curl_off_t /*dlNow*/, curl_off_t ulTotal,
                                  curl_off_t ulNow) {
    auto* ctx = static_cast<ProgressContext*>(clientp);
    if (ctx && ctx->callback && ulTotal > 0) {
        ctx->callback(static_cast<long long>(ulNow),
                      static_cast<long long>(ulTotal));
    }
    return 0;
}

UploadResult UploadFile(const std::string& filePath,
                        const std::string& serverUrl,
                        ProgressCallback onProgress) {
    UploadResult result{false, 0, "", ""};

    // Verify file exists.
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            result.errorMessage = "Failed to open file: " + filePath;
            return result;
        }
    }

    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();
    if (!curl) {
        result.errorMessage = "Failed to initialize curl";
        curl_global_cleanup();
        return result;
    }

    curl_mime* mime = curl_mime_init(curl);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "video");
    curl_mime_filedata(part, filePath.c_str());

    std::string responseBody;
    curl_easy_setopt(curl, CURLOPT_URL, serverUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "VideoUploader/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);

    ProgressContext progressCtx{onProgress};
    if (onProgress) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, CurlProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressCtx);
    }

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        result.errorMessage = curl_easy_strerror(res);
    } else {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        result.httpCode = httpCode;
        result.responseBody = responseBody;
        result.success = (httpCode >= 200 && httpCode < 300);
        if (!result.success) {
            result.errorMessage = "HTTP " + std::to_string(httpCode);
        }
    }

    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    return result;
}

} // namespace VideoUpload
