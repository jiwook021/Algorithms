/**
 * @file WebRtcLoop.cpp
 * @brief Implementation of the WebRTC looping video streamer.
 */

#include "WebRtcLoop.hpp"
#include <iostream>
#include <sstream>

namespace WebRtcLoop {

static GstElement* pipeline = nullptr;
static GstElement* webrtcbin = nullptr;
static GMainLoop*  mainLoop = nullptr;
static GstBus*     bus = nullptr;

// --- Callbacks ---

static void OnDemuxPadAdded(GstElement*, GstPad* pad, gpointer) {
    GstCaps* caps = gst_pad_get_current_caps(pad);
    const char* name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    gst_caps_unref(caps);

    if (g_str_has_prefix(name, "video/")) {
        GstElement* queue = gst_bin_get_by_name(GST_BIN(pipeline), "video_queue");
        GstPad* sink = gst_element_get_static_pad(queue, "sink");
        gst_pad_link(pad, sink);
        gst_object_unref(sink);
        gst_object_unref(queue);
    }
}

static void OnNegotiationNeeded(GstElement*, gpointer) {
    GstPromise* promise = gst_promise_new_with_change_func(
        [](GstPromise* p, gpointer) {
            GstWebRTCSessionDescription* offer = nullptr;
            const GstStructure* reply = gst_promise_get_reply(p);
            gst_structure_get(reply, "offer",
                              GST_TYPE_WEBRTC_SESSION_DESCRIPTION,
                              &offer, nullptr);
            gst_promise_unref(p);

            g_signal_emit_by_name(webrtcbin,
                                  "set-local-description", offer, nullptr);

            gchar* sdpText = gst_sdp_message_as_text(offer->sdp);
            std::cout << "\n=== SDP OFFER BEGIN ===\n"
                      << sdpText
                      << "=== SDP OFFER END ===\n";
            g_free(sdpText);
            gst_webrtc_session_description_free(offer);

            std::cout << "\nPaste SDP ANSWER, end with an empty line:\n";
            std::stringstream ss;
            std::string line;
            while (std::getline(std::cin, line) && !line.empty())
                ss << line << "\n";
            auto answerStr = ss.str();

            GstSDPMessage* sdpMsg = nullptr;
            gst_sdp_message_new(&sdpMsg);
            gst_sdp_message_parse_buffer(
                reinterpret_cast<const guint8*>(answerStr.c_str()),
                static_cast<guint>(answerStr.size()),
                sdpMsg);
            GstWebRTCSessionDescription* answer =
                gst_webrtc_session_description_new(
                    GST_WEBRTC_SDP_TYPE_ANSWER, sdpMsg);
            g_signal_emit_by_name(webrtcbin,
                                  "set-remote-description", answer, nullptr);
            gst_webrtc_session_description_free(answer);
        },
        nullptr, nullptr);
    g_signal_emit_by_name(webrtcbin, "create-offer", nullptr, promise);
}

static void OnIceCandidate(GstElement*, guint mline, gchar* cand, gpointer) {
    std::cout << "ICE CANDIDATE: " << mline << " " << cand << "\n";
}

static void OnEos(GstBus*, GstMessage*, gpointer) {
    GstSeekFlags flags = static_cast<GstSeekFlags>(
        GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT);
    gst_element_seek(pipeline, 1.0, GST_FORMAT_TIME,
                     flags,
                     GST_SEEK_TYPE_SET, 0,
                     GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);
}

// --- Public API ---

bool BuildPipeline(const std::string& filePath) {
    pipeline   = gst_pipeline_new("loop-pipeline");
    GstElement* filesrc = gst_element_factory_make("filesrc",    "src");
    GstElement* demux   = gst_element_factory_make("qtdemux",    "demux");
    GstElement* vqueue  = gst_element_factory_make("queue",      "video_queue");
    GstElement* pay     = gst_element_factory_make("rtph264pay", "pay");
    webrtcbin           = gst_element_factory_make("webrtcbin",  "webrtc");

    if (!pipeline || !filesrc || !demux || !vqueue || !pay || !webrtcbin) {
        std::cerr << "Missing GStreamer elements\n";
        return false;
    }

    g_object_set(filesrc, "location", filePath.c_str(), nullptr);
    g_object_set(pay, "config-interval", 1, "pt", 96, nullptr);
    gst_bin_add_many(GST_BIN(pipeline),
                     filesrc, demux, vqueue, pay, webrtcbin, nullptr);
    gst_element_link(filesrc, demux);
    gst_element_link(vqueue, pay);

    {
        GstPad* src  = gst_element_get_static_pad(pay, "src");
        GstPad* sink = gst_element_request_pad_simple(webrtcbin, "sink_%u");
        gst_pad_link(src, sink);
        gst_object_unref(src);
        gst_object_unref(sink);
    }

    g_signal_connect(demux,     "pad-added",
                     G_CALLBACK(OnDemuxPadAdded), nullptr);
    g_signal_connect(webrtcbin, "on-negotiation-needed",
                     G_CALLBACK(OnNegotiationNeeded), nullptr);
    g_signal_connect(webrtcbin, "on-ice-candidate",
                     G_CALLBACK(OnIceCandidate), nullptr);

    bus = gst_element_get_bus(pipeline);
    gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message::eos", G_CALLBACK(OnEos), nullptr);

    return true;
}

void Run() {
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    mainLoop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(mainLoop);
}

void Cleanup() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
    }
    if (bus) {
        gst_object_unref(bus);
        bus = nullptr;
    }
    if (pipeline) {
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }
    if (mainLoop) {
        g_main_loop_unref(mainLoop);
        mainLoop = nullptr;
    }
}

} // namespace WebRtcLoop
