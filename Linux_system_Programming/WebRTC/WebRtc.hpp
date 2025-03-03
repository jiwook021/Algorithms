#pragma once

/**
 * @file WebRtc.hpp
 * @brief WebRTC peer connection management using GStreamer.
 *
 * This module wraps GStreamer's webrtcbin element to provide a simplified
 * C++ interface for WebRTC session establishment: SDP offer/answer exchange,
 * ICE candidate gathering, and media pipeline management.
 *
 * NOTE: Building requires GStreamer development libraries:
 *   pkg-config --cflags --libs gstreamer-1.0 gstreamer-sdp-1.0 gstreamer-webrtc-1.0
 *
 * For unit-testing purposes, the module also exposes pure-logic helpers
 * (SDP type parsing, ICE candidate string validation) that can be tested
 * without GStreamer being installed.
 */

#include <string>
#include <vector>
#include <optional>

namespace WebRtc {

// ============================================================================
// SDP helpers (testable without GStreamer)
// ============================================================================

enum class SdpType { OFFER, ANSWER, UNKNOWN };

/// Convert a string like "offer" / "answer" to the SdpType enum.
SdpType ParseSdpType(const std::string& typeStr);

/// Convert SdpType back to a lowercase string.
std::string SdpTypeToString(SdpType type);

/// Minimal validation: an SDP blob must start with "v=0".
bool ValidateSdpBlob(const std::string& sdp);

// ============================================================================
// ICE candidate helpers (testable without GStreamer)
// ============================================================================

struct IceCandidate {
    int sdpMlineIndex;
    std::string candidate;
};

/// Parse an ICE candidate line (format: "<mline_index> <candidate_string>").
std::optional<IceCandidate> ParseIceCandidateLine(const std::string& line);

/// Check whether a candidate string looks structurally valid
/// (starts with "candidate:" or "a=candidate:").
bool ValidateIceCandidateString(const std::string& candidate);

// ============================================================================
// CLI command parsing (testable without GStreamer)
// ============================================================================

enum class CliCommand {
    START,
    CREATE_OFFER,
    STOP,
    SET_OFFER,
    SET_ANSWER,
    ADD_ICE,
    QUIT,
    UNKNOWN
};

/// Map a raw command string to a CliCommand enum.
CliCommand ParseCliCommand(const std::string& input);

} // namespace WebRtc
