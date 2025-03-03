#include <alsa/asoundlib.h>    // ALSA PCM API
#include <iostream>             // std::cout, std::cerr
#include <fstream>              // std::ofstream for WAV file
#include <vector>               // std::vector for audio buffer
#include <csignal>              // std::signal, std::sig_atomic_t for Ctrl+C
#include <cstring>              // std::memcpy
#include <cstdint>              // uint32_t, uint16_t

// WAV file header format (RIFF)
#pragma pack(push, 1)
struct WavHeader {
    char     Riff[4];        // "RIFF"
    uint32_t FileSize;       // Size of entire file minus 8 bytes
    char     Wave[4];        // "WAVE"
    char     Fmt[4];         // "fmt "
    uint32_t FmtSize;        // Length of format data (16 for PCM)
    uint16_t AudioFormat;    // Audio format (1 = PCM)
    uint16_t NumChannels;    // Number of channels
    uint32_t SampleRate;     // Sampling rate
    uint32_t ByteRate;       // sampleRate * numChannels * bitsPerSample/8
    uint16_t BlockAlign;     // numChannels * bitsPerSample/8
    uint16_t BitsPerSample;  // Bits per sample (e.g., 16)
    char     data[4];        // "data"
    uint32_t DataSize;       // Size of sample data in bytes
};
#pragma pack(pop)

// Configuration constants
constexpr char           PCM_DEVICE[]        = "default";           // ALSA capture device
constexpr unsigned int   SAMPLE_RATE         = 44100;               // Samples per second
constexpr snd_pcm_format_t FORMAT             = SND_PCM_FORMAT_S16_LE; // 16-bit LE samples
constexpr unsigned int   CHANNELS            = 1;                   // Mono
constexpr snd_pcm_uframes_t PERIOD_FRAMES    = 1024;                // Frames per ALSA period

// Signal flag for clean exit
static std::sig_atomic_t KeepRecording = 1;
void HandleSigInt(int) { KeepRecording = 0; }

int main() {
    std::signal(SIGINT, HandleSigInt);  // Handle Ctrl+C

    // Open output WAV file
    std::ofstream WavFile("recorded.wav", std::ios::binary);
    if (!WavFile) {
        std::cerr << "Failed to open output WAV file.\n";
        return 1;
    }

    // Prepare WAV header with placeholder sizes
    WavHeader Header;
    std::memcpy(Header.Riff, "RIFF", 4);
    Header.FileSize      = 0;  // to be updated
    std::memcpy(Header.Wave, "WAVE", 4);
    std::memcpy(Header.Fmt,  "fmt ", 4);
    Header.FmtSize        = 16;
    Header.AudioFormat    = 1;  // PCM
    Header.NumChannels    = CHANNELS;
    Header.SampleRate     = SAMPLE_RATE;
    Header.BitsPerSample  = snd_pcm_format_width(FORMAT);
    Header.ByteRate       = SAMPLE_RATE * CHANNELS * Header.BitsPerSample / 8;
    Header.BlockAlign     = CHANNELS * Header.BitsPerSample / 8;
    std::memcpy(Header.data, "data", 4);
    Header.DataSize       = 0;  // to be updated

    // Write placeholder header
    WavFile.write(reinterpret_cast<const char*>(&Header), sizeof(Header));

    // ALSA handles and parameters
    snd_pcm_t *PcmHandle = nullptr;
    snd_pcm_hw_params_t *HwParams = nullptr;
    int Err;

    // 1) Open PCM capture device (blocking)
    if ((Err = snd_pcm_open(&PcmHandle, PCM_DEVICE,
                            SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        std::cerr << "Error opening PCM device: " << snd_strerror(Err) << "\n";
        return 1;
    }

    // 2) Allocate hardware parameters object
    snd_pcm_hw_params_alloca(&HwParams);
    snd_pcm_hw_params_any(PcmHandle, HwParams);
    // Set interleaved mode
    snd_pcm_hw_params_set_access(PcmHandle, HwParams,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    // Set sample format
    snd_pcm_hw_params_set_format(PcmHandle, HwParams, FORMAT);
    // Set channels
    snd_pcm_hw_params_set_channels(PcmHandle, HwParams, CHANNELS);
    // Set rate
    unsigned int Rate = SAMPLE_RATE;
    int Dir = 0;
    snd_pcm_hw_params_set_rate_near(PcmHandle, HwParams, &Rate, &Dir);
    // Set period size
    snd_pcm_uframes_t Period = PERIOD_FRAMES;
    snd_pcm_hw_params_set_period_size_near(PcmHandle, HwParams,
                                           &Period, &Dir);
    // Apply parameters
    snd_pcm_hw_params(PcmHandle, HwParams);
    snd_pcm_prepare(PcmHandle);

    // Buffer for one period of samples (mono: one sample/frame)
    std::vector<int16_t> Buffer(PERIOD_FRAMES * CHANNELS);
    uint32_t TotalDataBytes = 0;

    std::cout << "Recording... Press Ctrl+C to stop.\n";

    // 3) Capture loop: read and write until interrupted
    while (KeepRecording) {
        snd_pcm_sframes_t Frames = snd_pcm_readi(PcmHandle,
                                                Buffer.data(),
                                                PERIOD_FRAMES);
        if (Frames == -EPIPE) {
            // Overrun: recover
            snd_pcm_prepare(PcmHandle);
            continue;
        } else if (Frames < 0) {
            std::cerr << "Read error: " << snd_strerror(Frames) << "\n";
            break;
        }
        // Write raw PCM data to WAV file
        std::size_t Bytes = Frames * CHANNELS * (Header.BitsPerSample / 8);
        WavFile.write(reinterpret_cast<const char*>(Buffer.data()), Bytes);
        TotalDataBytes += Bytes;
    }

    // 4) Update header with correct sizes
    // fileSize = 4 ("WAVE") + (8 + fmtSize) + (8 + dataSize)
    Header.DataSize = TotalDataBytes;
    Header.FileSize = 4 + (8 + Header.FmtSize) + (8 + Header.DataSize);

    // Seek to beginning and rewrite header
    WavFile.seekp(0, std::ios::beg);
    WavFile.write(reinterpret_cast<const char*>(&Header), sizeof(Header));
    WavFile.close();

    // Cleanup ALSA
    snd_pcm_close(PcmHandle);

    std::cout << "Recording saved to 'recorded.wav' ("
              << TotalDataBytes << " bytes of PCM data)" << std::endl;

    return 0;
}

/*
Time Complexity: O(N), where N = number of frames recorded.
Memory Complexity: O(P), where P = PERIOD_FRAMES (fixed buffer size).
*/
