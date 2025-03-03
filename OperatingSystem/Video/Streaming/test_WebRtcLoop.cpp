/**
 * @file test_WebRtcLoop.cpp
 * @brief Unit tests for WebRtcLoop module.
 *
 * GStreamer WebRTC requires a display / full media stack, so these tests
 * verify GStreamer initialization and basic element availability.
 */

#include <gst/gst.h>
#include <cassert>
#include <iostream>

/// GStreamer can be initialized.
void TestGstInit() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
    // If we get here, init succeeded.
    std::cout << "[PASS] TestGstInit\n";
}

/// Required GStreamer elements exist in the plugin registry.
void TestElementsExist() {
    auto* filesrc = gst_element_factory_make("filesrc", nullptr);
    assert(filesrc != nullptr);
    gst_object_unref(filesrc);

    auto* queue = gst_element_factory_make("queue", nullptr);
    assert(queue != nullptr);
    gst_object_unref(queue);

    std::cout << "[PASS] TestElementsExist\n";
}

/// Pipeline can be created and destroyed.
void TestPipelineLifecycle() {
    auto* pipeline = gst_pipeline_new("test-pipeline");
    assert(pipeline != nullptr);

    auto* src = gst_element_factory_make("fakesrc", "src");
    auto* sink = gst_element_factory_make("fakesink", "sink");
    assert(src && sink);

    gst_bin_add_many(GST_BIN(pipeline), src, sink, nullptr);
    gboolean linked = gst_element_link(src, sink);
    assert(linked);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    std::cout << "[PASS] TestPipelineLifecycle\n";
}

int main() {
    TestGstInit();
    TestElementsExist();
    TestPipelineLifecycle();
    std::cout << "\nAll WebRtcLoop tests passed.\n";
    return 0;
}
