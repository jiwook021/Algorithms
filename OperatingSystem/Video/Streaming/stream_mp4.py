#!/usr/bin/env python3
"""
stream_mp4.py — Stream and loop an MP4 file over WebRTC with GStreamer + asyncio.

Usage:
    python3 stream_mp4.py <path/to/video.mp4> ws://<signaling_host>:<port>

This version:
  • Waits 1 second after PLAYING to allow dynamic pads to link
  • Then manually triggers SDP offer, which will now include video/audio m= lines
  • Loops the MP4 file on EOS

Time complexity: O(1) per buffer (real‑time).  
Memory footprint: bounded by GStreamer’s queue sizes (~100 KB).
"""
import sys, os, asyncio, json, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import websockets

# Initialize GStreamer
Gst.init(None)

class WebRTCStreamer:
    def __init__(self, mp4_path: str, signaling_uri: str):
        if not os.path.isfile(mp4_path):
            raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
        self.mp4_path      = mp4_path
        self.signaling_uri = signaling_uri
        self.ws            = None

        # Build pipeline
        self.pipeline  = Gst.Pipeline.new("webrtc-streamer")
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtcbin")
        if not self.webrtcbin:
            raise RuntimeError("Could not create webrtcbin element")
        self.webrtcbin.set_property(
            "stun-server", "stun://stun.l.google.com:19302"
        )

        src   = Gst.ElementFactory.make("filesrc",  "file-source")
        demux = Gst.ElementFactory.make("qtdemux",  "qt-demux")
        src.set_property("location", mp4_path)

        for elem in (src, demux, self.webrtcbin):
            self.pipeline.add(elem)
        src.link(demux)
        demux.connect("pad-added", self._on_demux_pad)

        # Loop on EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos)

        # ICE candidates
        self.webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)

    def _on_demux_pad(self, demux, pad):
        caps = pad.get_current_caps().to_string()
        if caps.startswith("video/"):
            chain = ["queue", "decodebin", "videoconvert", "vp8enc", "rtpvp8pay"]
        else:
            chain = ["queue", "decodebin", "audioconvert", "audioresample", "opusenc", "rtpopuspay"]

        elems = []
        for name in chain:
            e = Gst.ElementFactory.make(name, None)
            if not e:
                raise RuntimeError(f"Failed to create element {name}")
            self.pipeline.add(e)
            e.sync_state_with_parent()
            elems.append(e)

        pad.link(elems[0].get_static_pad("sink"))
        elems[0].link(elems[1])
        elems[1].connect("pad-added",
                         lambda db, p: p.link(elems[2].get_static_pad("sink")))
        for a, b in zip(elems[2:], elems[3:]):
            a.link(b)
        elems[-1].link(self.webrtcbin)

    def _on_eos(self, bus, msg):
        print("[Streamer] ⚡ EOS reached, looping to start")
        self.pipeline.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            0
        )

    def _on_ice_candidate(self, webrtc, mline, candidate):
        msg = { "type":"ice", "sdpMLineIndex":mline, "candidate":candidate }
        asyncio.create_task(self.ws.send(json.dumps(msg)))
        print(f"[Streamer] → Sent ICE candidate (m={mline})")

    async def _receive_signals(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            t   = msg.get("type")
            if t == "answer":
                print("[Streamer] ← Received SDP answer")
                sdp = Gst.SDPMessage.new()
                Gst.SDPMessage.parse_buffer(msg["sdp"].encode(), sdp)
                answer = Gst.WebRTCSessionDescription.new(
                    Gst.WebRTCSDPType.ANSWER, sdp)
                self.webrtcbin.emit("set-remote-description", answer, None)
            elif t == "ice":
                print("[Streamer] ← Received ICE candidate")
                self.webrtcbin.emit("add-ice-candidate",
                                     msg["sdpMLineIndex"], msg["candidate"])

    async def run(self):
        print(f"[Streamer] 🔌 Connecting to {self.signaling_uri}")
        self.ws = await websockets.connect(self.signaling_uri)
        print("[Streamer] ✔️ Signaling connected")

        # Start receiving answers & ICE
        asyncio.create_task(self._receive_signals())

        # Start playback
        self.pipeline.set_state(Gst.State.PLAYING)
        print("[Streamer] ▶ Pipeline set to PLAYING")
        # Wait until the pipeline is prerolled & dynamic pads linked
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

        # Give GStreamer a moment to link decodebin → webrtcbin
        await asyncio.sleep(1.0)

        # Now create and send the SDP offer
        print("[Streamer] ▶ Creating SDP offer")
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, self.webrtcbin)
        self.webrtcbin.emit("create-offer", None, promise)

        # Keep the script alive
        await asyncio.Event().wait()

    def _on_offer_created(self, promise, webrtc):
        print("[Streamer] ▶ offer created")
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        if not offer:
            raise RuntimeError("SDP offer generation failed")
        webrtc.emit("set-local-description", offer, None)
        sdp_text = offer.sdp.as_text()
        asyncio.create_task(self.ws.send(json.dumps({ "type":"offer", "sdp":sdp_text })))
        print("[Streamer] → Sent SDP offer")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 stream_mp4.py <video.mp4> ws://<host>:<port>")
        sys.exit(1)

    mp4_file, ws_uri = sys.argv[1], sys.argv[2]
    streamer = WebRTCStreamer(mp4_file, ws_uri)
    try:
        asyncio.run(streamer.run())
    except KeyboardInterrupt:
        print("\n[Streamer] Interrupted — exiting.")
        streamer.pipeline.set_state(Gst.State.NULL)
        sys.exit(0)
