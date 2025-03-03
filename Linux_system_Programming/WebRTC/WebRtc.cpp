#include "WebRtc.hpp"
#include <sstream>
#include <algorithm>

namespace WebRtc {

// ============================================================================
// SDP helpers
// ============================================================================

SdpType ParseSdpType(const std::string& typeStr) {
    if (typeStr == "offer")  return SdpType::OFFER;
    if (typeStr == "answer") return SdpType::ANSWER;
    return SdpType::UNKNOWN;
}

std::string SdpTypeToString(SdpType type) {
    switch (type) {
        case SdpType::OFFER:  return "offer";
        case SdpType::ANSWER: return "answer";
        default:              return "unknown";
    }
}

bool ValidateSdpBlob(const std::string& sdp) {
    // Minimal check: SDP must start with the version line "v=0".
    if (sdp.empty()) return false;
    // Skip any leading whitespace / newlines.
    size_t pos = sdp.find_first_not_of(" \t\r\n");
    if (pos == std::string::npos) return false;
    return sdp.compare(pos, 3, "v=0") == 0;
}

// ============================================================================
// ICE candidate helpers
// ============================================================================

std::optional<IceCandidate> ParseIceCandidateLine(const std::string& line) {
    // Expected format: "<mline_index> <candidate_string>"
    std::istringstream iss(line);
    int mlineIndex;
    std::string candidate;

    if (!(iss >> mlineIndex)) return std::nullopt;
    // The rest of the line is the candidate string.
    std::getline(iss >> std::ws, candidate);
    if (candidate.empty()) return std::nullopt;

    return IceCandidate{mlineIndex, candidate};
}

bool ValidateIceCandidateString(const std::string& candidate) {
    if (candidate.empty()) return false;
    // Typical ICE candidate starts with "candidate:" or "a=candidate:"
    if (candidate.find("candidate:") != std::string::npos) return true;
    return false;
}

// ============================================================================
// CLI command parsing
// ============================================================================

CliCommand ParseCliCommand(const std::string& input) {
    // Trim leading/trailing whitespace.
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return CliCommand::UNKNOWN;
    size_t end = input.find_last_not_of(" \t\r\n");
    std::string trimmed = input.substr(start, end - start + 1);

    // Convert to lowercase for comparison.
    std::string lower = trimmed;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "start")        return CliCommand::START;
    if (lower == "create-offer") return CliCommand::CREATE_OFFER;
    if (lower == "stop")         return CliCommand::STOP;
    if (lower == "offer")        return CliCommand::SET_OFFER;
    if (lower == "answer")       return CliCommand::SET_ANSWER;
    if (lower.substr(0, 3) == "ice") return CliCommand::ADD_ICE;
    if (lower == "quit")         return CliCommand::QUIT;
    return CliCommand::UNKNOWN;
}

} // namespace WebRtc
