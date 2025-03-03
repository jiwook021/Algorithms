#include <alsa/asoundlib.h>   // ALSA library header
#include <cmath>              // For sine wave generation
#include <iostream>           // For std::cerr, std::cout

// ALSA PCM device name ("default" means system default audio output)
#define PCM_DEVICE "default"

// Audio parameters
static constexpr int SAMPLE_RATE = 44100;  // Sample rate in Hz
static constexpr int FREQUENCY = 440;  .0       // Frequency of the tone (A4 note)
static constexpr int DURATION = 2;  // Playback duration in seconds

int main() {
    snd_pcm_t *PcmHandle;               // PCM device handle
    snd_pcm_hw_params_t *Params;         // Hardware parameters structure
    int Dir;                             // Direction variable for ALSA functions
    int Rc;                              // Return code from ALSA functions

    // Open the PCM device in playback mode
    Rc = snd_pcm_open(&PcmHandle, PCM_DEVICE, SND_PCM_STREAM_PLAYBACK, 0);
    if (Rc < 0) {
        std::cerr << "Unable to open PCM device: " << snd_strerror(Rc) << "\n";
        return 1;
    }

    // Allocate and initialize hardware parameters structure
    snd_pcm_hw_params_alloca(&Params);
    snd_pcm_hw_params_any(PcmHandle, Params);  // Fill with default values

    // Set desired hardware parameters
    snd_pcm_hw_params_set_access(PcmHandle, Params, SND_PCM_ACCESS_RW_INTERLEAVED); // Interleaved mode
    snd_pcm_hw_params_set_format(PcmHandle, Params, SND_PCM_FORMAT_S16_LE);         // 16-bit little-endian
    snd_pcm_hw_params_set_channels(PcmHandle, Params, 2);                           // Stereo output (2 channels)

    unsigned int SampleRate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(PcmHandle, Params, &SampleRate, &Dir);         // Set sample rate

    // Apply hardware parameters to the PCM device
    snd_pcm_hw_params(PcmHandle, Params);

    // Set period size (number of frames per ALSA period)
    int Frames = 32;
    snd_pcm_hw_params_set_period_size_near(PcmHandle, Params, (snd_pcm_uframes_t *)&Frames, &Dir);

    // Prepare PCM device for use
    snd_pcm_prepare(PcmHandle);

    // Allocate buffer: stereo → 2 samples per frame
    int16_t Buffer[Frames * 2];

    // Variables for sine wave generation
    double Phase = 0.0;
    double Increment = 2.0 * M_PI * FREQUENCY / SAMPLE_RATE;

    int TotalFrames = SAMPLE_RATE * DURATION;

    // Generate and play audio data
    for (int i = 0; i < TotalFrames; i += Frames) {
        for (int f = 0; f < Frames; ++f) {
            int16_t Sample = static_cast<int16_t>(32767 * sin(Phase)); // Sine wave amplitude
            Buffer[2 * f] = Sample;       // Left channel
            Buffer[2 * f + 1] = Sample;   // Right channel
            Phase += Increment;
        }

        // Write buffer to PCM device
        snd_pcm_writei(PcmHandle, Buffer, Frames);
    }

    // Drain pending audio samples and close PCM device
    snd_pcm_drain(PcmHandle);
    snd_pcm_close(PcmHandle);

    return 0;
}
