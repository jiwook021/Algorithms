// client.js
const video   = document.getElementById("video");
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});

pc.ontrack = e => {
  video.srcObject = e.streams[0];
};

pc.onicecandidate = e => {
  if (e.candidate)
    console.log("ICE CANDIDATE:", JSON.stringify(e.candidate));
};

// 1) Paste the C++ SDP OFFER here as a string:
const offer = `v=0
o=- 4586472294161479139 0 IN IP4 0.0.0.0
s=-
t=0 0
a=ice-options:trickle
m=video 9 UDP/TLS/RTP/SAVPF 96
c=IN IP4 0.0.0.0
a=setup:actpass
a=ice-ufrag:KFwiMnVK7HUOeVyj41+Xjk17TDPpsJg+
a=ice-pwd:NKEdY4kEY373QWOLr+fg0Fy9CmjTUcOI
a=rtcp-mux
a=rtcp-rsize
a=sendrecv
a=rtpmap:96 H264/90000
a=rtcp-fb:96 nack pli
a=rtcp-fb:96 transport-cc
a=framerate:30
a=fmtp:96 packetization-mode=1;sprop-parameter-sets=Z2QAKKzZQHgCJ+XARAAAAwAEAAADAPA8YMZY,aOviSyLA;profile-level-id=640028;level-asymmetry-allowed=1
a=ssrc:1377136352 msid:user4038983294@host-45939db5 webrtctransceiver0
a=ssrc:1377136352 cname:user4038983294@host-45939db5
a=mid:video0
a=fingerprint:sha-256 0D:62:CF:A0:5D:28:04:33:2A:30:DD:F9:C2:F3:31:43:53:CD:44:9B:8B:6B:FB:3C:2A:B9:5B:AC:08:74:86:66
a=rtcp-mux-only`;

(async () => {
  await pc.setRemoteDescription({ type: "offer", sdp: offer });
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  console.log("=== SDP ANSWER BEGIN ===\n" + pc.localDescription.sdp + "=== SDP ANSWER END ===");
})();
