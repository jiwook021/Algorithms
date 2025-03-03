#include <alsa/asoundlib.h>   // ALSA PCM API
#include <iostream>
#include <vector>
#include <cstring>            // For std::memcpy

#define PCM_DEVICE "default"        // ALSA default PCM device (maps to PulseAudio on WSL)
static constexpr int SAMPLE_RATE = 44100;  // Sampling rate in Hz (CD quality)
static constexpr int CHANNELS = 1;  // Mono audio
static constexpr int SECONDS_TO_RECORD = 5;  // Duration of recording
#define FORMAT SND_PCM_FORMAT_S16_LE // 16-bit Little Endian audio

int main() {
    snd_pcm_t *PcmCaptureHandle, *PcmPlaybackHandle; // PCM device handles
    snd_pcm_hw_params_t *HwParams;                      // Hardware parameters object
    int Rc;

    // 1. Open PCM capture device (microphone)
    Rc = snd_pcm_open(&PcmCaptureHandle, PCM_DEVICE, SND_PCM_STREAM_CAPTURE, 0);
    if (Rc < 0) {
        std::cerr << "Unable to open capture device: " << snd_strerror(Rc) << "\n";
        return 1;
    }

    // 2. Open PCM playback device (speaker)
    Rc = snd_pcm_open(&PcmPlaybackHandle, PCM_DEVICE, SND_PCM_STREAM_PLAYBACK, 0);
    if (Rc < 0) {
        std::cerr << "Unable to open playback device: " << snd_strerror(Rc) << "\n";
        return 1;
    }

    // ----------------------------- CAPTURE DEVICE SETUP -----------------------------

    // 3. Allocate and initialize hardware parameter structure for capture
    snd_pcm_hw_params_alloca(&HwParams);                      // Allocate on stack
    snd_pcm_hw_params_any(PcmCaptureHandle, HwParams);      // Fill with default values
    snd_pcm_hw_params_set_access(PcmCaptureHandle, HwParams, SND_PCM_ACCESS_RW_INTERLEAVED); // interleaved mode
    snd_pcm_hw_params_set_format(PcmCaptureHandle, HwParams, FORMAT);                        // audio format
    snd_pcm_hw_params_set_channels(PcmCaptureHandle, HwParams, CHANNELS);                    // mono
    unsigned int Rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(PcmCaptureHandle, HwParams, &Rate, nullptr);             // set sample rate
    snd_pcm_hw_params(PcmCaptureHandle, HwParams);                                            // apply config
    snd_pcm_prepare(PcmCaptureHandle);                                                         // prepare device

    // ----------------------------- PLAYBACK DEVICE SETUP -----------------------------

    // 4. Allocate and configure hardware parameter structure for playback
    snd_pcm_hw_params_alloca(&HwParams);
    snd_pcm_hw_params_any(PcmPlaybackHandle, HwParams);  // Init with defaults
    snd_pcm_hw_params_set_access(PcmPlaybackHandle, HwParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(PcmPlaybackHandle, HwParams, FORMAT);
    snd_pcm_hw_params_set_channels(PcmPlaybackHandle, HwParams, CHANNELS);
    snd_pcm_hw_params_set_rate_near(PcmPlaybackHandle, HwParams, &Rate, nullptr);
    snd_pcm_hw_params(PcmPlaybackHandle, HwParams);
    snd_pcm_prepare(PcmPlaybackHandle);

    // ----------------------------- RECORDING -----------------------------

    // 5. Frame and buffer configuration
    const int FrameSize = snd_pcm_format_width(FORMAT) / 8 * CHANNELS; // Bytes per frame
    const int TotalFrames = SAMPLE_RATE * SECONDS_TO_RECORD;           // Total frames to record
    const int BufferFrames = 1024;                                     // Chunk per read/write
    const int BufferBytes = BufferFrames * FrameSize;                // Bytes per chunk

    std::vector<char> Buffer(BufferBytes);                             // Temporary I/O buffer
    std::vector<char> RecordedData(TotalFrames * FrameSize);         // Total recording buffer

    std::cout << "Recording for " << SECONDS_TO_RECORD << " seconds...\n";

    // 6. Read audio data from microphone into memory buffer
    int Recorded = 0;
    while (Recorded < TotalFrames) {
        int Frames = std::min(BufferFrames, TotalFrames - Recorded);

        // Read frames from capture device
        int Rc = snd_pcm_readi(PcmCaptureHandle, Buffer.data(), Frames);
        if (Rc < 0) {
            std::cerr << "Read error: " << snd_strerror(Rc) << "\n";
            snd_pcm_prepare(PcmCaptureHandle); // Re-prepare if overrun occurs
            continue;
        }

        // Copy recorded frames into full buffer
        std::memcpy(&RecordedData[Recorded * FrameSize], Buffer.data(), Rc * FrameSize);
        Recorded += Rc;
    }

    std::cout << "Recording complete. Playing back...\n";

    // ----------------------------- PLAYBACK -----------------------------

    // 7. Write recorded data to speaker
    int Played = 0;
    while (Played < TotalFrames) {
        int Frames = std::min(BufferFrames, TotalFrames - Played);

        // Copy chunk of audio into write buffer
        std::memcpy(Buffer.data(), &RecordedData[Played * FrameSize], Frames * FrameSize);

        // Write frames to playback device
        Rc = snd_pcm_writei(PcmPlaybackHandle, Buffer.data(), Frames);
        if (Rc < 0) {
            std::cerr << "Write error: " << snd_strerror(Rc) << "\n";
            snd_pcm_prepare(PcmPlaybackHandle); // Re-prepare if underrun occurs
            continue;
        }
        Played += Rc;
    }

    // ----------------------------- CLEANUP -----------------------------

    snd_pcm_drain(PcmPlaybackHandle);   // Wait for all remaining samples to play
    snd_pcm_close(PcmCaptureHandle);    // Close capture device
    snd_pcm_close(PcmPlaybackHandle);   // Close playback device

    std::cout << "Done.\n";
    return 0;
}
