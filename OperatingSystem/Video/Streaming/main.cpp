/**
 * @file main.cpp
 * @brief Driver for the WebRtcLoop module.
 */

#include "WebRtcLoop.hpp"
#include <gst/gst.h>
#include <iostream>

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <clip.mp4>\n";
        return 1;
    }

    if (!WebRtcLoop::BuildPipeline(argv[1])) {
        return 1;
    }

    WebRtcLoop::Run();
    WebRtcLoop::Cleanup();
    return 0;
}
