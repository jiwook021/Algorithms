#include <alsa/asoundlib.h>    // ALSA sound API
#include <fstream>             // For file input (ifstream)
#include <iostream>            // For std::cerr and std::cout

#define PCM_DEVICE "default"   // Default ALSA PCM playback device

// WAV file header structure (based on RIFF format)
struct WavHeader {
    char Riff[4];              // "RIFF"
    uint32_t OverallSize;     // Overall file size - 8 bytes
    char Wave[4];              // "WAVE"
    char FmtChunkMarker[4];  // "fmt "
    uint32_t LengthOfFmt;    // Length of format data (16 for PCM)
    uint16_t FormatType;      // Format type (1 for PCM)
    uint16_t Channels;         // Number of channels (1=Mono, 2=Stereo)
    uint32_t SampleRate;      // Sampling rate (e.g., 44100)
    uint32_t Byterate;         // Bytes per second
    uint16_t BlockAlign;      // Number of bytes per frame
    uint16_t BitsPerSample;  // Number of bits per sample (e.g., 16)
    char DataChunkHeader[4]; // "data"
    uint32_t DataSize;        // Number of bytes in data section
};

int main(int argc, char *argv[]) {
    // Check if WAV file path is given as argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file.wav\n";
        return 1;
    }

    // Open WAV file in binary mode
    std::ifstream File(argv[1], std::ios::binary);
    if (!File) {
        std::cerr << "Unable to open WAV file.\n";
        return 1;
    }

    // Read WAV header
    WavHeader Header;
    File.read(reinterpret_cast<char*>(&Header), sizeof(WavHeader));

    // Validate RIFF and WAVE identifiers
    if (std::string(Header.Riff, 4) != "RIFF" || std::string(Header.Wave, 4) != "WAVE") {
        std::cerr << "Invalid WAV file.\n";
        return 1;
    }

    // ALSA PCM handle
    snd_pcm_t *PcmHandle;

    // Open ALSA playback device
    snd_pcm_open(&PcmHandle, PCM_DEVICE, SND_PCM_STREAM_PLAYBACK, 0);

    // Allocate hardware parameter structure
    snd_pcm_hw_params_t *Params;
    snd_pcm_hw_params_alloca(&Params);

    // Fill with default values
    snd_pcm_hw_params_any(PcmHandle, Params);

    // Set interleaved access mode (LRLRLR...)
    snd_pcm_hw_params_set_access(PcmHandle, Params, SND_PCM_ACCESS_RW_INTERLEAVED);

    // Set audio format to 16-bit little-endian
    snd_pcm_hw_params_set_format(PcmHandle, Params, SND_PCM_FORMAT_S16_LE);

    // Set number of audio channels (1 = mono, 2 = stereo)
    snd_pcm_hw_params_set_channels(PcmHandle, Params, Header.Channels);

    // Set sample rate from WAV file
    unsigned int Rate = Header.SampleRate;
    snd_pcm_hw_params_set_rate_near(PcmHandle, Params, &Rate, nullptr);

    // Apply hardware settings to the PCM device
    snd_pcm_hw_params(PcmHandle, Params);

    // Prepare PCM device before writing data
    snd_pcm_prepare(PcmHandle);

    // Define buffer for audio playback
    constexpr size_t BufferSize = 4096;
    char Buffer[BufferSize];

    // Main loop: read audio data from file and write to ALSA
    while (File.read(Buffer, BufferSize) || File.gcount()) {
        // Compute number of frames (samples per channel)
        int Frames = File.gcount() / (Header.BitsPerSample / 8 * Header.Channels);

        // Send frames to ALSA for playback
        snd_pcm_writei(PcmHandle, Buffer, Frames);
    }

    // Wait for playback buffer to drain
    snd_pcm_drain(PcmHandle);

    // Close PCM device
    snd_pcm_close(PcmHandle);

    return 0;
}
