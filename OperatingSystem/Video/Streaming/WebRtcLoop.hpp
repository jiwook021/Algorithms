/**
 * @file WebRtcLoop.hpp
 * @brief WebRTC looping video streamer using GStreamer
 * @details Streams an H.264 video track from an MP4 file in a loop to a
 *          browser via WebRTC. Signaling is done via console SDP copy-paste.
 */

#pragma once

#include <gst/gst.h>
#include <gst/webrtc/webrtc.h>
#include <gst/sdp/gstsdpmessage.h>

#include <string>

namespace WebRtcLoop {

/// Build and return the GStreamer pipeline for looping WebRTC.
/// @param filePath  Path to the MP4 clip.
/// @return true on success, false if elements are missing.
bool BuildPipeline(const std::string& filePath);

/// Run the GLib main loop (blocks until quit).
void Run();

/// Clean up pipeline resources.
void Cleanup();

} // namespace WebRtcLoop
