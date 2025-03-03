/**
 * @file main.cpp
 * @brief Entry point for the WebRTC demo application.
 *
 * The full GStreamer-based WebRTC pipeline (FixedWebRTC class) lives in the
 * original fixed_webrtc.cpp and requires GStreamer dev libraries to compile.
 *
 * This driver demonstrates the testable helpers exposed by WebRtc.hpp and
 * provides a stub CLI loop that mirrors the original interactive interface.
 */

#include "WebRtc.hpp"
#include <iostream>
#include <string>

int main() {
    std::cout << "WebRTC Helper Demo (no GStreamer dependency)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Commands: start, create-offer, stop, offer, answer, ice <mline> <candidate>, quit\n"
              << std::endl;

    std::string line;
    while (true) {
        std::cout << "Enter command: ";
        if (!std::getline(std::cin, line)) break;

        auto cmd = WebRtc::ParseCliCommand(line);
        switch (cmd) {
            case WebRtc::CliCommand::START:
                std::cout << "[stub] Would start GStreamer pipeline." << std::endl;
                break;
            case WebRtc::CliCommand::CREATE_OFFER:
                std::cout << "[stub] Would create SDP offer." << std::endl;
                break;
            case WebRtc::CliCommand::STOP:
                std::cout << "[stub] Would stop pipeline." << std::endl;
                break;
            case WebRtc::CliCommand::SET_OFFER:
                std::cout << "[stub] Enter SDP offer (end with '.'):" << std::endl;
                {
                    std::string sdp;
                    std::string sdpLine;
                    while (std::getline(std::cin, sdpLine) && sdpLine != ".")
                        sdp += sdpLine + "\n";
                    bool valid = WebRtc::ValidateSdpBlob(sdp);
                    std::cout << "SDP valid: " << (valid ? "yes" : "no") << std::endl;
                }
                break;
            case WebRtc::CliCommand::SET_ANSWER:
                std::cout << "[stub] Enter SDP answer (end with '.'):" << std::endl;
                {
                    std::string sdp;
                    std::string sdpLine;
                    while (std::getline(std::cin, sdpLine) && sdpLine != ".")
                        sdp += sdpLine + "\n";
                    bool valid = WebRtc::ValidateSdpBlob(sdp);
                    std::cout << "SDP valid: " << (valid ? "yes" : "no") << std::endl;
                }
                break;
            case WebRtc::CliCommand::ADD_ICE: {
                // Extract the part after "ice "
                size_t pos = line.find(' ');
                if (pos != std::string::npos) {
                    auto parsed = WebRtc::ParseIceCandidateLine(line.substr(pos + 1));
                    if (parsed) {
                        std::cout << "[stub] Would add ICE candidate mline="
                                  << parsed->sdpMlineIndex << " candidate="
                                  << parsed->candidate << std::endl;
                    } else {
                        std::cout << "Invalid ICE format. Usage: ice <mline> <candidate>" << std::endl;
                    }
                }
                break;
            }
            case WebRtc::CliCommand::QUIT:
                std::cout << "Exiting." << std::endl;
                return 0;
            default:
                std::cout << "Unknown command: " << line << std::endl;
                break;
        }
    }

    return 0;
}
