/**
 * @file main.cpp
 * @brief Driver for FIR_IIR
 */

#include "FIR_IIR.hpp"

int main() {
    try {
        // -------- 1) Generate random signal --------
        constexpr size_t NUM_SAMPLES = 200;
        std::vector<double> Signal(NUM_SAMPLES);

        // seed random engine with high-resolution clock
        std::mt19937_64 Rng(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        // normal distribution with mean=0, stddev=1
        std::normal_distribution<double> Dist(0.0, 1.0);

        // fill `signal` using std::generate and the distribution
        std::generate(Signal.begin(), Signal.end(),
            [&]() { return Dist(Rng); }
        );

        if (Signal.empty()) {
            std::cerr << "Error: no samples generated\n";
            return EXIT_FAILURE;
        }

        // -------- 2) Set up filters --------
        // FIR taps: simple weighted moving average
        std::vector<double> FirTaps = { 0.1, 0.15, 0.5, 0.15, 0.1 };
        FIRFilter Fir(FirTaps);

        // IIR alpha: controls smoothing (0 < α < 1)
        IIRFilter Iir(0.1);

        // reserve space for outputs to avoid reallocations
        std::vector<double> FirOutput;
        std::vector<double> IirOutput;
        FirOutput.reserve(NUM_SAMPLES);
        IirOutput.reserve(NUM_SAMPLES);

        // apply both filters in one pass
        for (double Sample : Signal) {
            FirOutput.push_back(Fir.Process(Sample));
            IirOutput.push_back(Iir.Process(Sample));
        }

        // -------- 3) Determine scaling for plotting --------
        // find global min/max across all series
        double MinVal = *std::min_element(Signal.begin(), Signal.end());
        MinVal = std::min(MinVal, *std::min_element(FirOutput.begin(), FirOutput.end()));
        MinVal = std::min(MinVal, *std::min_element(IirOutput.begin(), IirOutput.end()));

        double MaxVal = *std::max_element(Signal.begin(), Signal.end());
        MaxVal = std::max(MaxVal, *std::max_element(FirOutput.begin(), FirOutput.end()));
        MaxVal = std::max(MaxVal, *std::max_element(IirOutput.begin(), IirOutput.end()));

        // avoid division by zero in degenerate case
        if (MaxVal == MinVal) {
            MaxVal += 1.0;
            MinVal -= 1.0;
        }

        // -------- 4) Prepare OpenCV image --------
        constexpr int IMG_WIDTH  = 800;
        constexpr int IMG_HEIGHT = 600;
        // create white background image: CV_8UC3 = 8-bit uchar, 3 channels (BGR)
        cv::Mat PlotImg(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, cv::Scalar(255,255,255));

        // lambda to map (index, value) -> cv::Point(x,y)
        auto ToPoint = [&](size_t Idx, double value) {
            int x = static_cast<int>((double(Idx) / (NUM_SAMPLES - 1)) * (IMG_WIDTH - 1));
            int y = static_cast<int>(
                (1.0 - (value - MinVal) / (MaxVal - MinVal)) * (IMG_HEIGHT - 1)
            );
            return cv::Point{x, y};
        };

        // -------- 5) Draw the three curves --------
        // prepare point vectors
        std::vector<cv::Point> PtsSignal, PtsFIR, PtsIIR;
        PtsSignal.reserve(NUM_SAMPLES);
        PtsFIR.reserve(NUM_SAMPLES);
        PtsIIR.reserve(NUM_SAMPLES);

        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            PtsSignal.push_back(ToPoint(i, Signal[i]));
            PtsFIR   .push_back(ToPoint(i, FirOutput[i]));
            PtsIIR   .push_back(ToPoint(i, IirOutput[i]));
        }

        // draw polylines: false=no closed shape, thickness=2 pixels
        cv::polylines(PlotImg, PtsSignal, false, cv::Scalar(255, 0,   0  ), 2);
        cv::polylines(PlotImg, PtsFIR,    false, cv::Scalar(  0, 255, 0  ), 2);
        cv::polylines(PlotImg, PtsIIR,    false, cv::Scalar(  0, 0,   255), 2);

        // -------- 6) Add legend --------
        int LegendX = 10, LegendY = 20;
        // draw text and color box for each series
        cv::putText(PlotImg, "Original Signal", {LegendX+20, LegendY+0},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        cv::rectangle(PlotImg, {LegendX, LegendY-10}, {LegendX+15, LegendY+5},
                      cv::Scalar(255, 0, 0), cv::FILLED);
        LegendY += 20;
        cv::putText(PlotImg, "FIR Filtered", {LegendX+20, LegendY+0},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::rectangle(PlotImg, {LegendX, LegendY-10}, {LegendX+15, LegendY+5},
                      cv::Scalar(0, 255, 0), cv::FILLED);
        LegendY += 20;
        cv::putText(PlotImg, "IIR Filtered", {LegendX+20, LegendY+0},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(PlotImg, {LegendX, LegendY-10}, {LegendX+15, LegendY+5},
                      cv::Scalar(0, 0, 255), cv::FILLED);

        // -------- 7) Display and wait --------
        cv::imshow("Signal vs. FIR vs. IIR", PlotImg); // create window and show image
        cv::waitKey(0);                               // wait indefinitely for keypress
        cv::destroyAllWindows();                      // close all OpenCV windows
    }
    catch (const std::exception& Ex) {
        // catch any thrown exceptions and report
        std::cerr << "Exception: " << Ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
