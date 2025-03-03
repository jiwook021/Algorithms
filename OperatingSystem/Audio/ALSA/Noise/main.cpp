// File: speex_denoise.cpp
// Build: g++ -std=c++23 -O2 speex_denoise.cpp -lsndfile -lspeexdsp -o speex_denoise
//
// Usage:
//   ./speex_denoise input.wav output.wav
//
// This program reads `input.wav`, applies SpeexDSP noise suppression,
// and writes the result to `output.wav`.
//
// Time complexity: O(N) where N = total number of samples.
// Memory complexity: O(FRAME_SIZE) for the internal processing buffer.

#include <speex/speex_preprocess.h>
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <string>

static constexpr int FRAME_SIZE    = 160;    // samples per frame (10ms @ 16kHz)
static constexpr int SAMPLE_RATE   = 16000;  // Hz

// Helper: print error and exit
void ExitWithError(const std::string& Msg) {
    std::cerr << "Error: " << Msg << std::endl;
    std::exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> <output.wav>\n";
        return EXIT_FAILURE;
    }
    const std::string Infile  = argv[1];
    const std::string Outfile = argv[2];

    // 1) Open input file
    SF_INFO Sfinfo{};
    SNDFILE* InFile = SfOpen(Infile.c_str(), SFM_READ, &Sfinfo);
    if (!InFile) ExitWithError("cannot open input file: " + Infile);

    // Only support mono 16kHz PCM
    if (Sfinfo.Channels != 1 || Sfinfo.Samplerate != SAMPLE_RATE) {
        SfClose(InFile);
        ExitWithError("input must be mono WAV @ 16kHz");
    }

    // 2) Prepare output file (same format)
    SNDFILE* OutFile = SfOpen(Outfile.c_str(),
                               SFM_WRITE,
                               &Sfinfo);
    if (!OutFile) {
        SfClose(InFile);
        ExitWithError("cannot open output file: " + Outfile);
    }

    // 3) Initialize Speex preprocessor state
    SpeexPreprocessState* DenoiseState =
        SpeexPreprocessStateInit(FRAME_SIZE, SAMPLE_RATE);
    if (!DenoiseState) {
        SfClose(InFile);
        SfClose(OutFile);
        ExitWithError("failed to init Speex preprocess state");
    }
    // Enable noise suppression and VAD
    int DenoiseLevel = -30; // dB of noise suppression (e.g., -30dB)
    SpeexPreprocessCtl(DenoiseState, SPEEX_PREPROCESS_SET_DENOISE, &DenoiseLevel);
    int EnableVAD = 1;
    SpeexPreprocessCtl(DenoiseState, SPEEX_PREPROCESS_SET_VAD, &EnableVAD);

    // 4) Process frames
    std::vector<short> Buffer(FRAME_SIZE);
    SfCountT ReadCount;
    while ((ReadCount = SfReadShort(InFile, Buffer.data(), FRAME_SIZE)) > 0) {
        if (ReadCount < FRAME_SIZE) {
            // zero-pad last frame
            std::fill(Buffer.begin() + ReadCount, Buffer.end(), 0);
        }
        // apply noise suppression in-place
        SpeexPreprocessRun(DenoiseState, Buffer.data());
        // write processed frame
        SfWriteShort(OutFile, Buffer.data(), FRAME_SIZE);
    }

    // 5) Cleanup
    SpeexPreprocessStateDestroy(DenoiseState);
    SfClose(InFile);
    SfClose(OutFile);

    std::cout << "Denoising complete: " << Infile << " → " << Outfile << "\n";
    return EXIT_SUCCESS;
}

//./main ~/soundFiles/ex1.wav