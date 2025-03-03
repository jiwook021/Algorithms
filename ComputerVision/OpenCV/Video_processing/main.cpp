#include "VideoProcessing.hpp"
#include <iostream>
int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "video.mp4";
    std::string output = (argc > 2) ? argv[2] : "output_video.avi";
    if (!VideoProcessing::ProcessVideo(input, output, VideoProcessing::ConvertToGrayBgr))
        std::cerr << "Error processing video\n";
    else
        std::cout << "Output saved to " << output << "\n";
    return 0;
}
