/**
 * @file gstreamer_video_player.cpp
 * @brief A simple GStreamer video player implementation in C++
 * 
 * This example demonstrates how to:
 * 1. Initialize GStreamer
 * 2. Create a basic video playback pipeline
 * 3. Handle state changes and messages from the GStreamer bus
 * 4. Properly clean up resources
 */

 #include <gst/gst.h>
 #include <iostream>
 #include <string>
 #include <memory>
 #include <stdexcept>
 
 class GstreamerVideoPlayer {
 private:
     GstElement* Pipeline;      // The main pipeline
     GstElement* Source;        // File source element
     GstElement* Decoder;       // Decoding element
     GstElement* Converter;     // Video conversion element
     GstElement* Sink;          // Video sink element
     GstBus* Bus;               // Message bus
     std::string Filepath;      // Path to the video file
 
 public:
     /**
      * @brief Constructor for GstreamerVideoPlayer
      * @param file_path Path to the video file to play
      */
     GstreamerVideoPlayer(const std::string& FilePath) : 
         Pipeline(nullptr),
         Source(nullptr),
         Decoder(nullptr),
         Converter(nullptr),
         Sink(nullptr),
         Bus(nullptr),
         Filepath(FilePath) {
         
         // Initialize all elements to nullptr for safety
     }
 
     /**
      * @brief Destructor - ensures proper cleanup of GStreamer resources
      */
     ~GstreamerVideoPlayer() {
         // Properly clean up all allocated resources
         if (Pipeline) {
             gst_element_set_state(Pipeline, GST_STATE_NULL);
             gst_object_unref(Pipeline);
         }
         
         // Note: We don't need to unref individual elements within the pipeline
         // as they are owned by the pipeline and will be cleaned up automatically
     }
 
     /**
      * @brief Initialize GStreamer and create the pipeline
      * @return true on success, false on failure
      */
     bool Initialize() {
         try {
             // Create the pipeline elements
             Pipeline = gst_pipeline_new("video-player");
             if (!Pipeline) {
                 throw std::runtime_error("Failed to create pipeline element");
             }
 
             // Create file source element
             Source = gst_element_factory_make("filesrc", "file-source");
             if (!Source) {
                 throw std::runtime_error("Failed to create filesrc element");
             }
             
             // Set the file path property on the source element
             g_object_set(G_OBJECT(Source), "location", Filepath.c_str(), NULL);
 
             // Create decodebin element - automatically selects decoders
             Decoder = gst_element_factory_make("decodebin", "decoder");
             if (!Decoder) {
                 throw std::runtime_error("Failed to create decodebin element");
             }
 
             // Create videoconvert element - ensures format compatibility
             Converter = gst_element_factory_make("videoconvert", "converter");
             if (!Converter) {
                 throw std::runtime_error("Failed to create videoconvert element");
             }
 
             // Create video sink element - displays the video
             Sink = gst_element_factory_make("autovideosink", "video-sink");
             if (!Sink) {
                 throw std::runtime_error("Failed to create autovideosink element");
             }
 
             // Add all elements to the pipeline
             gst_bin_add_many(GST_BIN(Pipeline), Source, Decoder, Converter, Sink, NULL);
 
             // Link elements that can be linked statically
             // Note: decoder->converter will be linked dynamically when the pad becomes available
             if (!gst_element_link(Source, Decoder)) {
                 throw std::runtime_error("Failed to link source and decoder elements");
             }
             
             if (!gst_element_link(Converter, Sink)) {
                 throw std::runtime_error("Failed to link converter and sink elements");
             }
 
             // Connect to the pad-added signal on the decoder
             g_signal_connect(Decoder, "pad-added", G_CALLBACK(OnPadAddedStatic), this);
 
             // Get the bus from the pipeline
             Bus = gst_element_get_bus(Pipeline);
             if (!Bus) {
                 throw std::runtime_error("Failed to get bus from pipeline");
             }
 
             return true;
 
         } catch (const std::exception& e) {
             std::cerr << "Initialization error: " << e.what() << std::endl;
             Cleanup();
             return false;
         }
     }
 
     /**
      * @brief Start playing the video
      * @return true on success, false on failure
      */
     bool Play() {
         if (!Pipeline) {
             std::cerr << "Pipeline not initialized" << std::endl;
             return false;
         }
 
         // Set pipeline to playing state
         GstStateChangeReturn Ret = gst_element_set_state(Pipeline, GST_STATE_PLAYING);
         if (Ret == GST_STATE_CHANGE_FAILURE) {
             std::cerr << "Failed to start playback" << std::endl;
             return false;
         }
 
         std::cout << "Video playback started" << std::endl;
         return true;
     }
 
     /**
      * @brief Main loop to process messages from the GStreamer bus
      * @return 0 on normal exit, error code otherwise
      */
     int RunMainLoop() {
         if (!Pipeline || !Bus) {
             std::cerr << "Pipeline or bus not initialized" << std::endl;
             return -1;
         }
 
         std::cout << "Entering main loop, press Ctrl+C to exit" << std::endl;
         
         GstMessage* Msg = nullptr;
         bool Running = true;
         int Ret = 0;
 
         while (Running) {
             // Wait for a message on the bus
             Msg = gst_bus_timed_pop_filtered(
                 Bus,
                 GST_CLOCK_TIME_NONE,
                 (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_STATE_CHANGED)
             );
 
             if (Msg) {
                 switch (GST_MESSAGE_TYPE(Msg)) {
                     case GST_MESSAGE_ERROR: {
                         GError* Err = nullptr;
                         gchar* DebugInfo = nullptr;
                         
                         gst_message_parse_error(Msg, &Err, &DebugInfo);
                         std::cerr << "Error received from element " << GST_OBJECT_NAME(Msg->src) 
                                   << ": " << Err->message << std::endl;
                         
                         if (DebugInfo) {
                             std::cerr << "Debugging information: " << DebugInfo << std::endl;
                             g_free(DebugInfo);
                         }
                         
                         g_error_free(Err);
                         Running = false;
                         Ret = -1;
                         break;
                     }
                     
                     case GST_MESSAGE_EOS:
                         std::cout << "End of stream reached" << std::endl;
                         Running = false;
                         break;
                         
                     case GST_MESSAGE_STATE_CHANGED:
                         // We're only interested in state-changed messages from the pipeline
                         if (GST_MESSAGE_SRC(Msg) == GST_OBJECT(Pipeline)) {
                             GstState OldState, NewState, PendingState;
                             gst_message_parse_state_changed(Msg, &OldState, &NewState, &PendingState);
                             
                             std::cout << "Pipeline state changed from " 
                                       << gst_element_state_get_name(OldState) << " to " 
                                       << gst_element_state_get_name(NewState) << std::endl;
                         }
                         break;
                         
                     default:
                         // We should not reach here
                         std::cerr << "Unexpected message received" << std::endl;
                         break;
                 }
                 
                 gst_message_unref(Msg);
             }
         }
 
         return Ret;
     }
 
     /**
      * @brief Clean up resources
      */
     void Cleanup() {
         if (Bus) {
             gst_object_unref(Bus);
             Bus = nullptr;
         }
         
         if (Pipeline) {
             gst_element_set_state(Pipeline, GST_STATE_NULL);
             gst_object_unref(Pipeline);
             Pipeline = nullptr;
         }
         
         // Individual elements are owned by the pipeline and don't need explicit cleanup
         Source = nullptr;
         Decoder = nullptr;
         Converter = nullptr;
         Sink = nullptr;
     }
 
 private:
     /**
      * @brief Static callback for pad-added signal from decoder
      * @param src The decoder element that emitted the signal
      * @param new_pad The new pad that was added to the decoder
      * @param data Pointer to the GstreamerVideoPlayer instance
      */
     static void OnPadAddedStatic(GstElement* Src, GstPad* NewPad, gpointer data) {
         // Forward to the instance method
         GstreamerVideoPlayer* Player = static_cast<GstreamerVideoPlayer*>(data);
         Player->OnPadAdded(Src, NewPad);
     }
 
     /**
      * @brief Instance method to handle pad-added signal from decoder
      * @param src The decoder element that emitted the signal
      * @param new_pad The new pad that was added to the decoder
      */
     void OnPadAdded(GstElement* Src, GstPad* NewPad) {
         // Get the sink pad from the converter
         GstPad* SinkPad = gst_element_get_static_pad(Converter, "sink");
         
         // Check if the pad is already linked
         if (gst_pad_is_linked(SinkPad)) {
             std::cout << "Converter sink pad is already linked. Ignoring." << std::endl;
             gst_object_unref(SinkPad);
             return;
         }
 
         // Check the new pad's type
         GstCaps* NewPadCaps = gst_pad_get_current_caps(NewPad);
         if (!NewPadCaps) {
             std::cerr << "Failed to get caps of the new pad" << std::endl;
             gst_object_unref(SinkPad);
             return;
         }
         
         // Get the name of the media type
         GstStructure* NewPadStruct = gst_caps_get_structure(NewPadCaps, 0);
         const gchar* NewPadType = gst_structure_get_name(NewPadStruct);
         
         // Only link if it's a video pad
         if (g_str_has_prefix(NewPadType, "video/x-raw")) {
             // Link the pads
             GstPadLinkReturn Ret = gst_pad_link(NewPad, SinkPad);
             if (GST_PAD_LINK_FAILED(Ret)) {
                 std::cerr << "Type is '" << NewPadType << "' but link failed" << std::endl;
             } else {
                 std::cout << "Link succeeded (type: " << NewPadType << ")" << std::endl;
             }
         } else {
             std::cout << "Ignoring pad with type '" << NewPadType << "'" << std::endl;
         }
         
         // Clean up
         gst_caps_unref(NewPadCaps);
         gst_object_unref(SinkPad);
     }
 };
 
 /**
  * @brief Main function
  * @param argc Number of command line arguments
  * @param argv Array of command line arguments
  * @return 0 on success, error code on failure
  */
 int main(int argc, char* argv[]) {
     // Check command line arguments
     if (argc != 2) {
         std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
         return -1;
     }
 
     // Initialize GStreamer
     gst_init(&argc, &argv);
     
     std::string FilePath = argv[1];
     std::cout << "Playing video file: " << FilePath << std::endl;
     
     // Create and initialize the video player
     GstreamerVideoPlayer Player(FilePath);
     
     if (!Player.Initialize()) {
         std::cerr << "Failed to initialize GStreamer player" << std::endl;
         return -1;
     }
     
     // Start playback
     if (!Player.Play()) {
         std::cerr << "Failed to start playback" << std::endl;
         return -1;
     }
     
     // Enter main loop
     int Ret = Player.RunMainLoop();
     
     // Clean up is handled by the destructor
     return Ret;
 }