#include <alsa/asoundlib.h>  // ALSA PCM API
#include <iostream>
#include <vector>
#include <cstring>           // For std::memcpy

// Configuration constants
constexpr char PCM_DEVICE[]       = "default";             // ALSA device name
constexpr unsigned int SAMPLE_RATE     = 44100;            // Sampling rate (Hz)
constexpr snd_pcm_format_t AUDIO_FORMAT = SND_PCM_FORMAT_S16_LE; // 16-bit little-endian
constexpr unsigned int CHANNELS       = 1;                 // Mono I/O
constexpr snd_pcm_uframes_t PERIOD_SIZE = 1024;            // Frames per period
constexpr unsigned int PERIODS        = 4;                 // Number of periods in buffer

int main() {
    snd_pcm_t *CaptureHandle = nullptr;
    snd_pcm_t *PlaybackHandle = nullptr;
    snd_pcm_hw_params_t *CaptureParams = nullptr;
    snd_pcm_hw_params_t *PlaybackParams = nullptr;
    int Err;

    // 1) Open capture device for recording (blocking mode)
    if ((Err = snd_pcm_open(&CaptureHandle,
                            PCM_DEVICE,
                            SND_PCM_STREAM_CAPTURE,
                            0)) < 0) {
        std::cerr << "Error opening capture device: " << snd_strerror(Err) << "\n";
        return 1;
    }

    // 2) Open playback device for playback (blocking mode)
    if ((Err = snd_pcm_open(&PlaybackHandle,
                            PCM_DEVICE,
                            SND_PCM_STREAM_PLAYBACK,
                            0)) < 0) {
        std::cerr << "Error opening playback device: " << snd_strerror(Err) << "\n";
        snd_pcm_close(CaptureHandle);
        return 1;
    }

    // 3) Allocate hardware-params structures on the stack
    snd_pcm_hw_params_alloca(&CaptureParams);
    snd_pcm_hw_params_alloca(&PlaybackParams);

    // --- CONFIGURE CAPTURE PARAMETERS ---
    snd_pcm_hw_params_any(CaptureHandle, CaptureParams);
    // a) Interleaved mode: samples for each channel are interleaved
    snd_pcm_hw_params_set_access(CaptureHandle, CaptureParams,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    // b) Audio format (16-bit little-endian)
    snd_pcm_hw_params_set_format(CaptureHandle, CaptureParams,
                                 AUDIO_FORMAT);
    // c) Mono channel
    snd_pcm_hw_params_set_channels(CaptureHandle, CaptureParams,
                                   CHANNELS);
    // d) Sampling rate (nearest supported)
    unsigned int Rate = SAMPLE_RATE;
    int Dir = 0;
    snd_pcm_hw_params_set_rate_near(CaptureHandle, CaptureParams,
                                    &Rate, &Dir);
    // e) Period size (frames per chunk)
    snd_pcm_uframes_t PeriodSize = PERIOD_SIZE;
    snd_pcm_hw_params_set_period_size_near(CaptureHandle, CaptureParams,
                                           &PeriodSize, &Dir);
    // f) Buffer size = period_size * PERIODS
    snd_pcm_uframes_t BufferSize = PeriodSize * PERIODS;
    snd_pcm_hw_params_set_buffer_size_near(CaptureHandle, CaptureParams,
                                           &BufferSize);
    // g) Apply capture parameters to device
    snd_pcm_hw_params(CaptureHandle, CaptureParams);
    // h) Prepare device for use (required before I/O)
    snd_pcm_prepare(CaptureHandle);

    // --- CONFIGURE PLAYBACK PARAMETERS (mirror capture) ---
    snd_pcm_hw_params_any(PlaybackHandle, PlaybackParams);
    snd_pcm_hw_params_set_access(PlaybackHandle, PlaybackParams,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(PlaybackHandle, PlaybackParams,
                                 AUDIO_FORMAT);
    snd_pcm_hw_params_set_channels(PlaybackHandle, PlaybackParams,
                                   CHANNELS);
    snd_pcm_hw_params_set_rate_near(PlaybackHandle, PlaybackParams,
                                    &Rate, &Dir);
    // reuse period_size and buffer_size variables
    snd_pcm_hw_params_set_period_size_near(PlaybackHandle, PlaybackParams,
                                           &PeriodSize, &Dir);
    snd_pcm_hw_params_set_buffer_size_near(PlaybackHandle, PlaybackParams,
                                           &BufferSize);
    snd_pcm_hw_params(PlaybackHandle, PlaybackParams);
    snd_pcm_prepare(PlaybackHandle);

    // 6) Allocate I/O buffer
    const size_t FrameBytes = (snd_pcm_format_width(AUDIO_FORMAT) / 8) * CHANNELS;
    const size_t BufBytes   = PeriodSize * FrameBytes;
    std::vector<char> Buffer(BufBytes);

    std::cout << "Starting real-time loopback (Ctrl+C to stop)...\n";

    // 7) Main loop: capture → immediate playback
    while (true) {
        // Read exactly one period of frames
        snd_pcm_sframes_t FramesRead = snd_pcm_readi(CaptureHandle,
                                                      Buffer.data(),
                                                      PeriodSize);
        if (FramesRead == -EPIPE) {
            // Overrun: input buffer ran out of data
            std::cerr << "Capture overrun. Recovering...\n";
            snd_pcm_prepare(CaptureHandle);
            continue;
        } else if (FramesRead < 0) {
            std::cerr << "Capture error: " << snd_strerror(FramesRead) << "\n";
            break;
        }

        // Write those frames immediately to playback
        snd_pcm_sframes_t FramesWritten = snd_pcm_writei(PlaybackHandle,
                                                          Buffer.data(),
                                                          FramesRead);
        if (FramesWritten == -EPIPE) {
            // Underrun: output buffer starved
            std::cerr << "Playback underrun. Recovering...\n";
            snd_pcm_prepare(PlaybackHandle);
            continue;
        } else if (FramesWritten < 0) {
            std::cerr << "Playback error: " << snd_strerror(FramesWritten) << "\n";
            break;
        }
    }

    // 8) Cleanup
    snd_pcm_drop(CaptureHandle);     // Stop capture immediately
    snd_pcm_drain(PlaybackHandle);   // Play remaining samples
    snd_pcm_close(CaptureHandle);    // Close capture device
    snd_pcm_close(PlaybackHandle);   // Close playback device

    return 0;
}
