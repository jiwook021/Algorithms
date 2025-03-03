#include <alsa/asoundlib.h>   // ALSA PCM API: provides functions to open, configure, read, and write PCM audio devices
#include <iostream>             // std::cout, std::cerr for console I/O
#include <vector>               // std::vector to hold audio samples buffer
#include <cmath>                // std::sqrt(), std::log10() for RMS and dB calculations
#include <csignal>              // std::signal, std::sig_atomic_t for handling Ctrl+C
#include <limits>               // std::numeric_limits for normalizing sample values
#include <iomanip>              // std::fixed, std::setprecision for formatting dB output

// Configuration constants control sample rate, format, and display properties
constexpr char PCM_DEVICE[]      = "default";              // PCM device name: maps to the default capture source (e.g., microphone)
constexpr unsigned int SAMPLE_RATE   = 44100;              // Number of samples per second
constexpr snd_pcm_format_t FORMAT    = SND_PCM_FORMAT_S16_LE; // Each sample is 16 bits, little-endian
constexpr unsigned int CHANNELS      = 1;                  // Mono input: one channel per frame
constexpr snd_pcm_uframes_t PERIOD_SIZE = 1024;            // Frames per ALSA period read
constexpr unsigned int METER_WIDTH    = 50;                // Width of the console VU meter in characters

// Global flag to exit the main loop cleanly when Ctrl+C is pressed
static std::sig_atomic_t KeepRunning = 1;
// Signal handler: sets keepRunning to 0 when SIGINT (Ctrl+C) is received
void HandleSigInt(int) { KeepRunning = 0; }

int main() {
    // Register the signal handler for SIGINT
    std::signal(SIGINT, HandleSigInt);

    // ALSA capture device handle and hardware parameters pointer
    snd_pcm_t *CaptureHandle = nullptr;
    snd_pcm_hw_params_t *HwParams = nullptr;
    int Err;

    // 1) Open the PCM capture device in blocking mode
    //    - SND_PCM_STREAM_CAPTURE: for recording
    //    - 0 flags: default behavior (blocking)
    if ((Err = snd_pcm_open(&CaptureHandle,
                            PCM_DEVICE,
                            SND_PCM_STREAM_CAPTURE,
                            0)) < 0) {
        std::cerr << "Error opening capture device: " << snd_strerror(Err) << "\n";
        return 1;
    }

    // 2) Allocate a hardware-parameters object on the stack
    //    - snd_pcm_hw_params_alloca allocates using alloca() for automatic cleanup
    snd_pcm_hw_params_alloca(&HwParams);
    //    - Initialize hwParams with full hardware configuration space
    snd_pcm_hw_params_any(CaptureHandle, HwParams);

    // 3) Configure the hardware parameters:
    //    a) Access type: interleaved (samples in sequence per frame)
    snd_pcm_hw_params_set_access(CaptureHandle, HwParams,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    //    b) Sample format: 16-bit signed little-endian
    snd_pcm_hw_params_set_format(CaptureHandle, HwParams, FORMAT);
    //    c) Channel count: mono (1 channel per frame)
    snd_pcm_hw_params_set_channels(CaptureHandle, HwParams, CHANNELS);
    //    d) Sampling rate: set nearest available rate to SAMPLE_RATE
    unsigned int Rate = SAMPLE_RATE;
    int Dir = 0;  // Direction indicator for nearest rate adjustment
    snd_pcm_hw_params_set_rate_near(CaptureHandle, HwParams, &Rate, &Dir);
    //    e) Period size: number of frames per read operation
    snd_pcm_uframes_t Period = PERIOD_SIZE;
    snd_pcm_hw_params_set_period_size_near(CaptureHandle, HwParams,
                                           &Period, &Dir);

    // 4) Apply the configured parameters to the device and prepare it
    snd_pcm_hw_params(CaptureHandle, HwParams);   // Apply settings
    snd_pcm_prepare(CaptureHandle);               // Put device into a ready state

    // 5) Allocate a container for one period of audio data
    //    - Period size × channels samples, each sample is int16_t
    std::vector<int16_t> Buffer(PERIOD_SIZE * CHANNELS);

    std::cout << "VU Meter started—press Ctrl+C to stop\n";

    // 6) Calculate normalization factor: max absolute sample value
    const double MaxSample = static_cast<double>(std::numeric_limits<int16_t>::max());

    // 7) Main capture loop: runs until keepRunning is set to 0
    while (KeepRunning) {
        // a) Read exactly PERIOD_SIZE frames (mono → PERIOD_SIZE samples)
        snd_pcm_sframes_t FramesRead =
            snd_pcm_readi(CaptureHandle, Buffer.data(), PERIOD_SIZE);

        // b) Handle errors or overruns
        if (FramesRead == -EPIPE) {
            // Overrun: the ring buffer overflowed
            std::cerr << "\nOverrun detected. Recovering...\n";
            snd_pcm_prepare(CaptureHandle);  // Reset and prepare the device
            continue;
        } else if (FramesRead < 0) {
            // Other errors
            std::cerr << "\nRead error: " << snd_strerror(FramesRead) << "\n";
            break;
        }

        // c) Compute RMS (root mean square) for perceived loudness
        double SumSquares = 0.0;
        for (snd_pcm_sframes_t i = 0; i < FramesRead; ++i) {
            double Normalized = Buffer[i] / MaxSample;
            SumSquares += Normalized * Normalized;
        }
        double Rms = std::sqrt(SumSquares / FramesRead);          // Range: 0.0 .. 1.0
        double DB  = 20.0 * std::log10(Rms + 1e-12);              // Convert RMS to dB, avoid log(0)

        // d) Map RMS to number of meter bars (0 .. METER_WIDTH)
        int BarCount = static_cast<int>(Rms * METER_WIDTH + 0.5);

        // e) Render the VU meter: a bar of '█' and spaces
        std::cout << "\r[";
        for (unsigned int i = 0; i < METER_WIDTH; ++i) {
            std::cout << (i < BarCount ? "█" : " ");
        }
        // f) Show dB value formatted to one decimal place
        std::cout << "] " << std::fixed << std::setprecision(1)
                  << DB << " dB   " << std::flush;
    }

    // 8) Cleanup: close the ALSA capture device
    std::cout << "\nStopping VU Meter...\n";
    snd_pcm_close(CaptureHandle);
    return 0;
}
