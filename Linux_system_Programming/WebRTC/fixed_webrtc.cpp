#include <gst/gst.h>
#include <gst/Sdp/Sdp.h>
#include <gst/Webrtc/Webrtc.h>
#include <string>
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

// Class to handle WebRTC connection and signaling
class FixedWebRTC {
private:
    // GStreamer pipeline and elements
    GstElement* Pipeline_;
    GstElement* Webrtcbin_;
    
    // State variables
    bool PipelineRunning_;
    
    // Mutex for thread safety
    std::mutex Mutex_;
    
public:
    FixedWebRTC() : 
        Pipeline_(nullptr),
        Webrtcbin_(nullptr),
        PipelineRunning_(false) {
    }

    ~FixedWebRTC() {
        // Clean up GStreamer resources
        Stop();
    }

    /**
     * Initialize GStreamer and prepare for WebRTC
     * @return true if initialization is successful, false otherwise
     */
    bool Initialize() {
        // Initialize GStreamer
        GError* Error = nullptr;
        if (!GstInitCheck(nullptr, nullptr, &Error)) {
            std::cerr << "Failed to initialize GStreamer: " << Error->Message << std::endl;
            GErrorFree(Error);
            return false;
        }
        
        std::cout << "GStreamer initialized successfully" << std::endl;
        return true;
    }

    /**
     * Create a simplified GStreamer pipeline for WebRTC
     * @return true if pipeline creation is successful, false otherwise
     */
    bool CreatePipeline() {
        try {
            // Create a simpler pipeline string - this uses the pipeline syntax
            // instead of programmatically creating elements
            std::string PipelineStr = 
                "videotestsrc pattern=ball is-live=true ! videoconvert ! queue ! "
                "vp8enc deadline=1 error-resilient=partitions keyframe-max-dist=10 auto-alt-ref=true cpu-used=5 ! rtpvp8pay ! "
                "queue ! " 
                "application/x-rtp,media=video,encoding-name=VP8,payload=96 ! "
                "webrtcbin name=webrtcbin bundle-policy=max-bundle "
                "stun-server=stun://stun.l.google.com:19302 "
                "audiotestsrc wave=red-noise is-live=true ! audioconvert ! audioresample ! queue ! "
                "opusenc bitrate=48000 ! rtpopuspay ! "
                "queue ! "
                "application/x-rtp,media=audio,encoding-name=OPUS,payload=97 ! webrtcbin.";

            // Create pipeline from string
            GError* Error = NULL;
            Pipeline_ = GstParseLaunch(PipelineStr.c_str(), &Error);
            
            if (Error) {
                std::cerr << "Failed to create pipeline: " << Error->Message << std::endl;
                GErrorFree(Error);
                return false;
            }
            
            if (!Pipeline_) {
                std::cerr << "Failed to create pipeline (null pipeline)" << std::endl;
                return false;
            }
            
            // Get the webrtcbin element
            Webrtcbin_ = GstBinGetByName(GST_BIN(Pipeline_), "webrtcbin");
            if (!Webrtcbin_) {
                std::cerr << "Failed to find webrtcbin element" << std::endl;
                return false;
            }
            
            // Connect to WebRTC signals - IMPORTANT: NOT connecting to 'on-negotiation-needed'
            // to prevent automatic offer generation
                
            GSignalConnect(Webrtcbin_, "on-ice-candidate", 
                G_CALLBACK(+[](GstElement* Webrtcbin, Guint MlineIndex, Gchar* Candidate, Gpointer UserData) {
                    ((FixedWebRTC*)UserData)->OnIceCandidate(Webrtcbin, MlineIndex, Candidate);
                }), this);
                
            GSignalConnect(Webrtcbin_, "pad-added", 
                G_CALLBACK(+[](GstElement* Webrtcbin, GstPad* Pad, Gpointer UserData) {
                    ((FixedWebRTC*)UserData)->OnPadAdded(Webrtcbin, Pad);
                }), this);
            
            std::cout << "Pipeline created successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error creating pipeline: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * Manually create an offer (instead of relying on automatic negotiation)
     */
    void CreateOffer() {
        if (!Webrtcbin_) {
            std::cerr << "WebRTC element not available" << std::endl;
            return;
        }
        
        std::cout << "Manually creating offer..." << std::endl;
        
        // Create an offer
        GstPromise* promise = GstPromiseNewWithChangeFunc(
            +[](GstPromise* promise, Gpointer UserData) {
                ((FixedWebRTC*)UserData)->OnOfferCreated(promise);
            }, this, nullptr);
        
        GSignalEmitByName(Webrtcbin_, "create-offer", nullptr, promise);
    }

    /**
     * Start the WebRTC session
     * @return true if session starts successfully, false otherwise
     */
    bool Start() {
        if (!Pipeline_) {
            std::cerr << "Pipeline not created, cannot start" << std::endl;
            return false;
        }
        
        if (PipelineRunning_) {
            std::cerr << "Pipeline already running" << std::endl;
            return true;
        }
        
        // Set pipeline to PLAYING state
        GstStateChangeReturn Ret = GstElementSetState(Pipeline_, GST_STATE_PLAYING);
        if (Ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start pipeline" << std::endl;
            return false;
        }
        
        PipelineRunning_ = true;
        std::cout << "WebRTC session started" << std::endl;
        return true;
    }

    /**
     * Stop the WebRTC session and release resources
     */
    void Stop() {
        std::lock_guard<std::mutex> Lock(Mutex_);
        
        if (PipelineRunning_ && Pipeline_) {
            GstElementSetState(Pipeline_, GST_STATE_NULL);
            PipelineRunning_ = false;
            std::cout << "WebRTC session stopped" << std::endl;
        }
        
        if (Webrtcbin_) {
            GstObjectUnref(Webrtcbin_);
            Webrtcbin_ = nullptr;
        }
        
        if (Pipeline_) {
            GstObjectUnref(Pipeline_);
            Pipeline_ = nullptr;
        }
    }

    /**
     * Set the remote SDP (Session Description Protocol) received from the peer
     * @param type: SDP type (offer or answer)
     * @param sdp_str: SDP string received from the remote peer
     * @return true if successful, false otherwise
     */
    bool SetRemoteSdp(const std::string& Type, const std::string& SdpStr) {
        if (!Webrtcbin_) {
            std::cerr << "WebRTC element not available" << std::endl;
            return false;
        }
        
        GstWebRTCSessionDescription* SessionDesc = nullptr;
        GstSDPMessage* SdpMsg = nullptr;
        
        // Parse the SDP string
        int Ret = GstSdpMessageNew(&SdpMsg);
        if (Ret != GST_SDP_OK) {
            std::cerr << "Failed to create SDP message" << std::endl;
            return false;
        }
        
        Ret = GstSdpMessageParseBuffer((const Guint8*)SdpStr.c_str(), SdpStr.length(), SdpMsg);
        if (Ret != GST_SDP_OK) {
            std::cerr << "Failed to parse SDP message" << std::endl;
            GstSdpMessageFree(SdpMsg);
            return false;
        }
        
        // Create the appropriate session description type
        GstWebRTCSDPType SdpType;
        if (Type == "offer") {
            SdpType = GST_WEBRTC_SDP_TYPE_OFFER;
        } else if (Type == "answer") {
            SdpType = GST_WEBRTC_SDP_TYPE_ANSWER;
        } else {
            std::cerr << "Unknown SDP type: " << Type << std::endl;
            GstSdpMessageFree(SdpMsg);
            return false;
        }
        
        SessionDesc = GstWebrtcSessionDescriptionNew(SdpType, SdpMsg);
        
        // Set remote description
        GstPromise* promise = GstPromiseNew();
        GSignalEmitByName(Webrtcbin_, "set-remote-description", SessionDesc, promise);
        
        // Wait for the promise to be resolved
        GstPromiseResult Result = GstPromiseWait(promise);
        if (Result != GST_PROMISE_RESULT_REPLIED) {
            std::cerr << "Error setting remote description: promise not replied" << std::endl;
            GstPromiseUnref(promise);
            GstWebrtcSessionDescriptionFree(SessionDesc);
            return false;
        }
        
        // Check for errors in the reply
        const GstStructure* Reply = GstPromiseGetReply(promise);
        if (Reply && GstStructureHasField(Reply, "error")) {
            const GError* Error = nullptr;
            GstStructureGet(Reply, "error", G_TYPE_ERROR, &Error, NULL);
            if (Error) {
                std::cerr << "Error setting remote description: " << Error->Message << std::endl;
                GErrorFree((GError*)Error);
                GstPromiseUnref(promise);
                GstWebrtcSessionDescriptionFree(SessionDesc);
                return false;
            }
        }
        
        // Clean up
        GstPromiseUnref(promise);
        GstWebrtcSessionDescriptionFree(SessionDesc);
        
        std::cout << "Remote " << Type << " SDP set successfully" << std::endl;
        
        // If we received an offer, create an answer
        if (Type == "offer") {
            // Create answer asynchronously
            promise = GstPromiseNewWithChangeFunc(
                +[](GstPromise* promise, Gpointer UserData) {
                    ((FixedWebRTC*)UserData)->OnAnswerCreated(promise);
                }, this, nullptr);
            
            GSignalEmitByName(Webrtcbin_, "create-answer", nullptr, promise);
        }
        
        return true;
    }

    /**
     * Add a remote ICE candidate received from peer
     * @param candidate: ICE candidate string
     * @param sdp_mline_index: SDP m-line index
     * @return true if successful, false otherwise
     */
    bool AddIceCandidate(const std::string& Candidate, int SdpMlineIndex) {
        if (!Webrtcbin_) {
            std::cerr << "WebRTC element not available" << std::endl;
            return false;
        }
        
        // Add ICE candidate to the webrtcbin element
        GSignalEmitByName(Webrtcbin_, "add-ice-candidate", SdpMlineIndex, Candidate.c_str());
        std::cout << "Added remote ICE candidate (mline " << SdpMlineIndex << "): " << Candidate << std::endl;
        return true;
    }

private:
    /**
     * Handler for when a new pad is added to the WebRTC element
     * @param webrtcbin: The WebRTC element
     * @param pad: The new pad
     */
    void OnPadAdded(GstElement* Webrtcbin, GstPad* Pad) {
        std::string PadName = GST_PAD_NAME(Pad);
        std::cout << "Pad added: " << PadName << std::endl;
        
        // For a simple test, we don't need to handle incoming audio/video
        // In a real application, we would link these pads to appropriate sinks
    }

    /**
     * Handler for when an ICE candidate is generated
     * @param webrtcbin: The WebRTC element
     * @param mline_index: SDP m-line index
     * @param candidate: ICE candidate string
     */
    void OnIceCandidate(GstElement* Webrtcbin, Guint MlineIndex, Gchar* Candidate) {
        std::cout << "Local ICE candidate (mline " << MlineIndex << "): " << Candidate << std::endl;
        
        // In a real application, we would send this candidate to the peer
        // through a signaling mechanism
    }

    /**
     * Handler for when an offer is created
     * @param promise: The GstPromise containing the offer
     */
    void OnOfferCreated(GstPromise* promise) {
        const GstStructure* Reply = GstPromiseGetReply(promise);
        GstWebRTCSessionDescription* Offer = nullptr;
        
        if (!Reply) {
            std::cerr << "Empty promise reply for offer creation" << std::endl;
            GstPromiseUnref(promise);
            return;
        }
        
        if (!GstStructureGet(Reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &Offer, nullptr)) {
            std::cerr << "Failed to get offer from promise" << std::endl;
            GstPromiseUnref(promise);
            return;
        }
        
        // Set local description
        GstPromise* LocalPromise = GstPromiseNew();
        GSignalEmitByName(Webrtcbin_, "set-local-description", Offer, LocalPromise);
        
        // Wait for the promise to be resolved
        GstPromiseResult Result = GstPromiseWait(LocalPromise);
        if (Result != GST_PROMISE_RESULT_REPLIED) {
            std::cerr << "Error setting local description: promise not replied" << std::endl;
            GstPromiseUnref(LocalPromise);
            GstWebrtcSessionDescriptionFree(Offer);
            GstPromiseUnref(promise);
            return;
        }
        
        // Check for errors in the reply
        const GstStructure* LocalReply = GstPromiseGetReply(LocalPromise);
        if (LocalReply && GstStructureHasField(LocalReply, "error")) {
            const GError* Error = nullptr;
            GstStructureGet(LocalReply, "error", G_TYPE_ERROR, &Error, NULL);
            if (Error) {
                std::cerr << "Error setting local description: " << Error->Message << std::endl;
                GErrorFree((GError*)Error);
            }
        }
        
        GstPromiseUnref(LocalPromise);
        
        // Print the SDP for easy copying
        std::string SdpStr(GstSdpMessageAsText(Offer->Sdp));
        std::cout << "\nLocal SDP (offer) - Copy this to the browser:\n" 
                  << "-------------------------------------------\n"
                  << SdpStr 
                  << "\n-------------------------------------------\n" << std::endl;
        
        GstWebrtcSessionDescriptionFree(Offer);
        GstPromiseUnref(promise);
    }

    /**
     * Handler for when an answer is created
     * @param promise: The GstPromise containing the answer
     */
    void OnAnswerCreated(GstPromise* promise) {
        const GstStructure* Reply = GstPromiseGetReply(promise);
        GstWebRTCSessionDescription* Answer = nullptr;
        
        if (!Reply) {
            std::cerr << "Empty promise reply for answer creation" << std::endl;
            GstPromiseUnref(promise);
            return;
        }
        
        if (!GstStructureGet(Reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &Answer, nullptr)) {
            std::cerr << "Failed to get answer from promise" << std::endl;
            GstPromiseUnref(promise);
            return;
        }
        
        // Set local description
        GstPromise* LocalPromise = GstPromiseNew();
        GSignalEmitByName(Webrtcbin_, "set-local-description", Answer, LocalPromise);
        
        // Wait for the promise to be resolved
        GstPromiseResult Result = GstPromiseWait(LocalPromise);
        if (Result != GST_PROMISE_RESULT_REPLIED) {
            std::cerr << "Error setting local description: promise not replied" << std::endl;
            GstPromiseUnref(LocalPromise);
            GstWebrtcSessionDescriptionFree(Answer);
            GstPromiseUnref(promise);
            return;
        }
        
        // Check for errors in the reply
        const GstStructure* LocalReply = GstPromiseGetReply(LocalPromise);
        if (LocalReply && GstStructureHasField(LocalReply, "error")) {
            const GError* Error = nullptr;
            GstStructureGet(LocalReply, "error", G_TYPE_ERROR, &Error, NULL);
            if (Error) {
                std::cerr << "Error setting local description: " << Error->Message << std::endl;
                GErrorFree((GError*)Error);
            }
        }
        
        GstPromiseUnref(LocalPromise);
        
        // Print the SDP for easy copying
        std::string SdpStr(GstSdpMessageAsText(Answer->Sdp));
        std::cout << "\nLocal SDP (answer) - Copy this to the browser:\n" 
                  << "-------------------------------------------\n"
                  << SdpStr 
                  << "\n-------------------------------------------\n" << std::endl;
        
        GstWebrtcSessionDescriptionFree(Answer);
        GstPromiseUnref(promise);
    }
};

/**
 * Simple command-line interface to interact with the WebRTC connection
 * @param webrtc: Reference to the FixedWebRTC object
 */
void RunCli(FixedWebRTC& Webrtc) {
    std::cout << "\nWebRTC with Manual Negotiation CLI\n";
    std::cout << "==================================\n";
    std::cout << "Commands:\n";
    std::cout << "  start        - Start the WebRTC session\n";
    std::cout << "  create-offer - Manually create an SDP offer\n";
    std::cout << "  stop         - Stop the WebRTC session\n";
    std::cout << "  offer        - Set remote SDP as offer (end with a line containing just '.')\n";
    std::cout << "  answer       - Set remote SDP as answer (end with a line containing just '.')\n";
    std::cout << "  ice <mline> <candidate> - Add remote ICE candidate\n";
    std::cout << "  quit         - Exit the program\n";
    
    std::string Command, Line;
    while (true) {
        std::cout << "\nEnter command: ";
        std::getline(std::cin, Command);
        
        if (Command == "start") {
            if (Webrtc.Start()) {
                std::cout << "Session started successfully" << std::endl;
            } else {
                std::cerr << "Failed to start session" << std::endl;
            }
        } else if (Command == "create-offer") {
            Webrtc.CreateOffer();
        } else if (Command == "stop") {
            Webrtc.Stop();
            std::cout << "Session stopped" << std::endl;
        } else if (Command == "offer") {
            std::cout << "Enter remote offer SDP (end with a line containing just '.'): " << std::endl;
            std::string Sdp;
            while (std::getline(std::cin, Line) && Line != ".") {
                Sdp += Line + "\n";
            }
            
            if (Webrtc.SetRemoteSdp("offer", Sdp)) {
                std::cout << "Remote offer SDP set successfully" << std::endl;
            } else {
                std::cerr << "Failed to set remote offer SDP" << std::endl;
            }
        } else if (Command == "answer") {
            std::cout << "Enter remote answer SDP (end with a line containing just '.'): " << std::endl;
            std::string Sdp;
            while (std::getline(std::cin, Line) && Line != ".") {
                Sdp += Line + "\n";
            }
            
            if (Webrtc.SetRemoteSdp("answer", Sdp)) {
                std::cout << "Remote answer SDP set successfully" << std::endl;
            } else {
                std::cerr << "Failed to set remote answer SDP" << std::endl;
            }
        } else if (Command.substr(0, 3) == "ice") {
            std::string MlineStr;
            std::string Candidate;
            
            // Format: ice <mline> <candidate>
            size_t Pos = Command.find(' ', 4);
            if (Pos != std::string::npos) {
                MlineStr = Command.substr(4, Pos - 4);
                Candidate = Command.substr(Pos + 1);
                
                try {
                    int MlineIndex = std::stoi(MlineStr);
                    if (Webrtc.AddIceCandidate(Candidate, MlineIndex)) {
                        std::cout << "ICE candidate added successfully" << std::endl;
                    } else {
                        std::cerr << "Failed to add ICE candidate" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Invalid mline index: " << MlineStr << std::endl;
                }
            } else {
                std::cout << "Usage: ice <mline_index> <candidate_string>" << std::endl;
            }
        } else if (Command == "quit") {
            break;
        } else {
            std::cout << "Unknown command: " << Command << std::endl;
        }
    }
}

/**
 * Main function
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char** argv) {
    std::cout << "WebRTC with Manual Negotiation (GStreamer)" << std::endl;
    
    try {
        // Create WebRTC object
        FixedWebRTC Webrtc;
        
        // Initialize GStreamer
        if (!Webrtc.Initialize()) {
            throw std::runtime_error("Failed to initialize GStreamer");
        }
        
        // Create pipeline
        if (!Webrtc.CreatePipeline()) {
            throw std::runtime_error("Failed to create pipeline");
        }
        
        // Run the command-line interface
        RunCli(Webrtc);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}