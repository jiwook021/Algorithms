#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// Matrix class with bounds checking
class Matrix {
private:
    int rows, cols;
    std::vector<float> data;
    
    // Check if indices are valid
    bool isValidIndex(int i, int j) const {
        return i >= 0 && i < rows && j >= 0 && j < cols;
    }

public: 
    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0f) {
        std::cout << "Creating Matrix of size " << rows << "x" << cols << std::endl;
    }
    
    float& operator()(int i, int j) {
        if (!isValidIndex(i, j)) {
            std::cerr << "ERROR: Matrix access out of bounds! Requested (" << i << "," << j 
                      << ") but matrix size is " << rows << "x" << cols << std::endl;
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }
    
    const float& operator()(int i, int j) const {
        if (!isValidIndex(i, j)) {
            std::cerr << "ERROR: Matrix const access out of bounds! Requested (" << i << "," << j 
                      << ") but matrix size is " << rows << "x" << cols << std::endl;
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    
    // Initialize with random values (for weights)
    void randomize(float min = -1.0f, float max = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        
        for (auto& val : data) {
            val = dist(gen);
        }
    }
};

// Activation functions
float relu(float x) {
    return std::max(0.0f, x);
}

float reluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Convolutional Layer with debug output
class ConvLayer {
private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    std::vector<Matrix> kernels;
    std::vector<float> biases;

public:
    ConvLayer(int inputChannels, int outputChannels, int kernelSize) 
        : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize) {
        
        std::cout << "Creating ConvLayer: " << inputChannels << " input channels, " 
                  << outputChannels << " output channels, " << kernelSize << "x" << kernelSize << " kernel" << std::endl;
        
        // Initialize kernels with random weights
        for (int i = 0; i < outputChannels; i++) {
            for (int j = 0; j < inputChannels; j++) {
                Matrix kernel(kernelSize, kernelSize);
                kernel.randomize(-0.1f, 0.1f);
                kernels.push_back(kernel);
            }
        }
        
        // Initialize biases to zero
        biases.resize(outputChannels, 0.0f);
    }
    
    std::vector<Matrix> forward(const std::vector<Matrix>& input) {
        if (input.empty()) {
            std::cerr << "ERROR: Empty input to ConvLayer!" << std::endl;
            throw std::runtime_error("Empty input to ConvLayer");
        }
        
        std::cout << "ConvLayer forward: Processing " << input.size() << " input channels of size " 
                  << input[0].getRows() << "x" << input[0].getCols() << std::endl;
        
        // Check input dimensions match what we expect
        if (input.size() != inputChannels) {
            std::cerr << "ERROR: Expected " << inputChannels << " input channels, but got " 
                      << input.size() << std::endl;
            throw std::runtime_error("Input channel count mismatch");
        }
        
        // Calculate output dimensions
        int outHeight = input[0].getRows() - kernelSize + 1;
        int outWidth = input[0].getCols() - kernelSize + 1;
        
        if (outHeight <= 0 || outWidth <= 0) {
            std::cerr << "ERROR: Invalid output dimensions: " << outHeight << "x" << outWidth 
                      << " (input: " << input[0].getRows() << "x" << input[0].getCols() 
                      << ", kernel: " << kernelSize << "x" << kernelSize << ")" << std::endl;
            throw std::runtime_error("Invalid convolution output dimensions");
        }
        
        std::cout << "ConvLayer output will be " << outputChannels << " channels of size " 
                  << outHeight << "x" << outWidth << std::endl;
        
        // Create output matrices
        std::vector<Matrix> output(outputChannels, Matrix(outHeight, outWidth));
        
        // For each output channel
        for (int outCh = 0; outCh < outputChannels; outCh++) {
            // For each input channel
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                int kernelIdx = outCh * inputChannels + inCh;
                
                // Apply convolution
                for (int i = 0; i < outHeight; i++) {
                    for (int j = 0; j < outWidth; j++) {
                        float sum = 0.0f;
                        
                        // Apply kernel
                        for (int ki = 0; ki < kernelSize; ki++) {
                            for (int kj = 0; kj < kernelSize; kj++) {
                                sum += input[inCh](i + ki, j + kj) * kernels[kernelIdx](ki, kj);
                            }
                        }
                        
                        output[outCh](i, j) += sum;
                    }
                }
            }
            
            // Add bias and apply activation function
            for (int i = 0; i < outHeight; i++) {
                for (int j = 0; j < outWidth; j++) {
                    output[outCh](i, j) = relu(output[outCh](i, j) + biases[outCh]);
                }
            }
        }
        
        std::cout << "ConvLayer forward complete" << std::endl;
        return output;
    }
};

// Max Pooling Layer with debug output
class MaxPoolLayer {
private:
    int poolSize;

public:
    MaxPoolLayer(int poolSize) : poolSize(poolSize) {
        std::cout << "Creating MaxPoolLayer with pool size " << poolSize << "x" << poolSize << std::endl;
    }
    
    std::vector<Matrix> forward(const std::vector<Matrix>& input) {
        if (input.empty()) {
            std::cerr << "ERROR: Empty input to MaxPoolLayer!" << std::endl;
            throw std::runtime_error("Empty input to MaxPoolLayer");
        }
        
        std::cout << "MaxPoolLayer forward: Processing " << input.size() << " input channels of size " 
                  << input[0].getRows() << "x" << input[0].getCols() << std::endl;
        
        std::vector<Matrix> output;
        
        for (size_t ch = 0; ch < input.size(); ch++) {
            const auto& channel = input[ch];
            
            // Check if dimensions are divisible by poolSize
            if (channel.getRows() % poolSize != 0 || channel.getCols() % poolSize != 0) {
                std::cerr << "WARNING: Input dimensions (" << channel.getRows() << "x" << channel.getCols() 
                          << ") not divisible by pool size " << poolSize << std::endl;
            }
            
            int outRows = channel.getRows() / poolSize;
            int outCols = channel.getCols() / poolSize;
            
            std::cout << "Channel " << ch << " pooling output will be " << outRows << "x" << outCols << std::endl;
            
            Matrix pooled(outRows, outCols);
            
            for (int i = 0; i < outRows; i++) {
                for (int j = 0; j < outCols; j++) {
                    float maxVal = -std::numeric_limits<float>::max();
                    
                    // Find maximum in the pool region
                    for (int pi = 0; pi < poolSize; pi++) {
                        for (int pj = 0; pj < poolSize; pj++) {
                            // Check if we're within bounds of the input
                            int ri = i * poolSize + pi;
                            int cj = j * poolSize + pj;
                            
                            if (ri < channel.getRows() && cj < channel.getCols()) {
                                float val = channel(ri, cj);
                                maxVal = std::max(maxVal, val);
                            }
                        }
                    }
                    
                    pooled(i, j) = maxVal;
                }
            }
            
            output.push_back(pooled);
        }
        
        std::cout << "MaxPoolLayer forward complete" << std::endl;
        return output;
    }
};

// Fully Connected Layer with debug output
class FCLayer {
private:
    int inputSize;
    int outputSize;
    Matrix weights;
    std::vector<float> biases;

public:
    FCLayer(int inputSize, int outputSize) 
        : inputSize(inputSize), outputSize(outputSize), weights(inputSize, outputSize), biases(outputSize, 0.0f) {
        
        std::cout << "Creating FCLayer: " << inputSize << " inputs, " << outputSize << " outputs" << std::endl;
        weights.randomize(-0.1f, 0.1f);  // Initialize with small random weights
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        std::cout << "FCLayer forward: Processing " << input.size() << " inputs to produce " 
                  << outputSize << " outputs" << std::endl;
        
        // Check input size
        if (input.size() != inputSize) {
            std::cerr << "ERROR: FCLayer expected " << inputSize << " inputs, but got " 
                      << input.size() << std::endl;
            throw std::runtime_error("Input size mismatch in FCLayer");
        }
        
        std::vector<float> output(outputSize, 0.0f);
        
        // Compute output = weights * input + bias
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                output[j] += input[i] * weights(i, j);
            }
            output[j] += biases[j];
            // Apply ReLU activation
            output[j] = relu(output[j]);
        }
        
        std::cout << "FCLayer forward complete" << std::endl;
        return output;
    }
};

// Softmax activation for output layer
std::vector<float> softmax(const std::vector<float>& input) {
    if (input.empty()) {
        std::cerr << "ERROR: Empty input to softmax!" << std::endl;
        throw std::runtime_error("Empty input to softmax");
    }
    
    std::cout << "Applying softmax to " << input.size() << " values" << std::endl;
    
    std::vector<float> output(input.size());
    float maxVal = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    
    // Compute exp(x - max) for numerical stability
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    // Check for division by zero
    if (sum == 0.0f) {
        std::cerr << "ERROR: Softmax sum is zero!" << std::endl;
        throw std::runtime_error("Softmax division by zero");
    }
    
    // Normalize
    for (auto& val : output) {
        val /= sum;
    }
    
    return output;
}

// Simple CNN model for image classification
class SimpleCNN {
private:
    ConvLayer conv1;
    MaxPoolLayer pool1;
    ConvLayer conv2;
    MaxPoolLayer pool2;
    FCLayer fc1;
    FCLayer fc2;
    int numClasses;
    int imageWidth;
    int imageHeight;

public:
    SimpleCNN(int imageWidth, int imageHeight, int numClasses)
        : imageWidth(imageWidth), imageHeight(imageHeight),
          conv1(3, 16, 3),         // 3 input channels (RGB), 16 output channels, 3x3 kernel
          pool1(2),                // 2x2 pooling
          conv2(16, 32, 3),        // 16 input channels, 32 output channels, 3x3 kernel
          pool2(2),                // 2x2 pooling
          // The key calculation that might be causing issues:
          fc1(32 * ((imageWidth / 4 - 2) / 2) * ((imageHeight / 4 - 2) / 2), 128),  // Flattened size to 128
          fc2(128, numClasses),    // 128 to number of classes
          numClasses(numClasses) {
              
        std::cout << "SimpleCNN created with the following architecture:" << std::endl;
        std::cout << "Input size: " << imageWidth << "x" << imageHeight << "x3 (RGB)" << std::endl;
        std::cout << "Conv1: 16 filters of size 3x3, output size " << (imageWidth - 2) << "x" << (imageHeight - 2) << "x16" << std::endl;
        std::cout << "Pool1: 2x2 pooling, output size " << (imageWidth - 2) / 2 << "x" << (imageHeight - 2) / 2 << "x16" << std::endl;
        std::cout << "Conv2: 32 filters of size 3x3, output size " << ((imageWidth - 2) / 2 - 2) << "x" << ((imageHeight - 2) / 2 - 2) << "x32" << std::endl;
        std::cout << "Pool2: 2x2 pooling, output size " << ((imageWidth - 2) / 2 - 2) / 2 << "x" << ((imageHeight - 2) / 2 - 2) / 2 << "x32" << std::endl;
        
        int flattenedSize = 32 * ((imageWidth / 4 - 2) / 2) * ((imageHeight / 4 - 2) / 2);
        std::cout << "Flattened size: " << flattenedSize << std::endl;
        std::cout << "FC1: " << flattenedSize << " -> 128" << std::endl;
        std::cout << "FC2: 128 -> " << numClasses << std::endl;
    }
    
    std::vector<float> forward(const cv::Mat& image) {
        std::cout << "SimpleCNN forward: Starting forward pass on image of size " 
                  << image.cols << "x" << image.rows << std::endl;
        
        if (image.empty() || image.rows != imageHeight || image.cols != imageWidth) {
            std::cerr << "ERROR: Image dimensions mismatch! Expected " << imageWidth << "x" << imageHeight 
                      << " but got " << image.cols << "x" << image.rows << std::endl;
            throw std::runtime_error("Image dimensions mismatch");
        }
        
        // Convert OpenCV image to our format (vector of Matrix)
        std::vector<Matrix> input;
        
        // Create a matrix for each color channel (R, G, B)
        Matrix redChannel(image.rows, image.cols);
        Matrix greenChannel(image.rows, image.cols);
        Matrix blueChannel(image.rows, image.cols);
        
        std::cout << "Converting OpenCV Mat to internal format..." << std::endl;
        
        // Copy data from OpenCV Mat to our matrices
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                blueChannel(i, j) = pixel[0] / 255.0f;  // Normalize to [0, 1]
                greenChannel(i, j) = pixel[1] / 255.0f;
                redChannel(i, j) = pixel[2] / 255.0f;
            }
        }
        
        input.push_back(redChannel);
        input.push_back(greenChannel);
        input.push_back(blueChannel);
        
        std::cout << "Starting forward pass through network layers..." << std::endl;
        
        // Forward pass through the network
        std::cout << "Conv1 forward pass..." << std::endl;
        std::vector<Matrix> conv1Out = conv1.forward(input);
        
        std::cout << "Pool1 forward pass..." << std::endl;
        std::vector<Matrix> pool1Out = pool1.forward(conv1Out);
        
        std::cout << "Conv2 forward pass..." << std::endl;
        std::vector<Matrix> conv2Out = conv2.forward(pool1Out);
        
        std::cout << "Pool2 forward pass..." << std::endl;
        std::vector<Matrix> pool2Out = pool2.forward(conv2Out);
        
        // Flatten the output of the last pooling layer
        std::cout << "Flattening output with " << pool2Out.size() << " channels..." << std::endl;
        std::vector<float> flattened;
        
        for (size_t ch = 0; ch < pool2Out.size(); ch++) {
            const auto& channel = pool2Out[ch];
            std::cout << "Channel " << ch << " dimensions: " << channel.getRows() << "x" << channel.getCols() << std::endl;
            
            for (int i = 0; i < channel.getRows(); i++) {
                for (int j = 0; j < channel.getCols(); j++) {
                    flattened.push_back(channel(i, j));
                }
            }
        }
        
        std::cout << "Flattened size: " << flattened.size() << std::endl;
        
        // Pass through fully connected layers
        std::cout << "FC1 forward pass..." << std::endl;
        std::vector<float> fc1Out = fc1.forward(flattened);
        
        std::cout << "FC2 forward pass..." << std::endl;
        std::vector<float> fc2Out = fc2.forward(fc1Out);
        
        // Apply softmax to get class probabilities
        std::cout << "Softmax activation..." << std::endl;
        return softmax(fc2Out);
    }
    
    // Predict class from image
    int predict(const cv::Mat& image) {
        std::cout << "Starting prediction..." << std::endl;
        std::vector<float> probabilities = forward(image);
        
        // Find the class with highest probability
        auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
        int predictedClass = std::distance(probabilities.begin(), maxIt);
        
        std::cout << "Predicted class: " << predictedClass << " with probability " << *maxIt << std::endl;
        return predictedClass;
    }
};

// Image utilities class
class ImageUtils {
public:
    // Create a synthetic test image
    static cv::Mat createTestImage(int width, int height) {
        std::cout << "Creating synthetic test image of size " << width << "x" << height << std::endl;
        cv::Mat testImage(height, width, CV_8UC3);
        
        // Create a simple pattern (gradient and circle)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Create a gradient background
                uchar r = static_cast<uchar>(255 * x / width);
                uchar g = static_cast<uchar>(255 * y / height);
                uchar b = static_cast<uchar>(128);
                
                // Add a circle in the center
                float centerX = width / 2.0f;
                float centerY = height / 2.0f;
                float distance = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                
                if (distance < width / 4) {
                    r = 255;
                    g = 0;
                    b = 0;
                }
                
                testImage.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
        
        return testImage;
    }
    
    // Check if an image is valid and debug print its properties
    static bool isValidImage(const cv::Mat& image, const std::string& label) {
        if (image.empty()) {
            std::cerr << label << " is empty!" << std::endl;
            return false;
        }
        
        std::cout << label << " dimensions: " << image.cols << "x" << image.rows 
                  << " channels: " << image.channels() 
                  << " depth: " << image.depth()
                  << " type: " << image.type() << std::endl;
        
        // Check if image has expected number of channels
        if (image.channels() != 3) {
            std::cerr << "WARNING: " << label << " expected to have 3 channels, but has " 
                      << image.channels() << std::endl;
        }
        
        // Check if image data is continuous
        if (!image.isContinuous()) {
            std::cerr << "WARNING: " << label << " data is not continuous!" << std::endl;
        }
        
        return true;
    }
};

// Main function with comprehensive debugging
int main() {
    try {
        std::cout << "Starting CNN demo program..." << std::endl;
        
        // Define image dimensions and number of classes
        const int imageWidth = 32;
        const int imageHeight = 32;
        const int numClasses = 10;
        
        std::cout << "Creating CNN model..." << std::endl;
        // Create model
        SimpleCNN model(imageWidth, imageHeight, numClasses);
        
        // Try to load image from file
        std::cout << "Attempting to load image from file..." << std::endl;
        cv::Mat image;
        
        try {
            image = cv::imread("input.jpg", cv::IMREAD_COLOR);
            
            // Extensive image validation
            if (!ImageUtils::isValidImage(image, "Loaded image")) {
                std::cerr << "Warning: Could not read the image file or image is invalid." << std::endl;
                std::cout << "Using a synthetic test image instead." << std::endl;
                
                // Create a synthetic test image
                image = ImageUtils::createTestImage(imageWidth, imageHeight);
                ImageUtils::isValidImage(image, "Synthetic image");
            } else {
                std::cout << "Image loaded successfully. Original dimensions: " 
                          << image.cols << "x" << image.rows << std::endl;
                
                // Only resize if necessary
                if (image.cols != imageWidth || image.rows != imageHeight) {
                    std::cout << "Resizing image to " << imageWidth << "x" << imageHeight << std::endl;
                    
                    // Create a new matrix for the resized image
                    cv::Mat resizedImage;
                    cv::resize(image, resizedImage, cv::Size(imageWidth, imageHeight));
                    
                    // Check if resize was successful
                    if (!ImageUtils::isValidImage(resizedImage, "Resized image")) {
                        throw std::runtime_error("Image resize failed");
                    }
                    
                    // Assign the resized image
                    image = resizedImage;
                } else {
                    std::cout << "Image already has correct dimensions, skipping resize." << std::endl;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV exception: " << e.what() << std::endl;
            std::cout << "Using a synthetic test image instead." << std::endl;
            image = ImageUtils::createTestImage(imageWidth, imageHeight);
            ImageUtils::isValidImage(image, "Synthetic image (after exception)");
        }
        
        // Final check to make sure we have a valid image
        if (image.empty() || image.rows != imageHeight || image.cols != imageWidth) {
            throw std::runtime_error("Failed to create a valid image for processing");
        }
        
        // Make prediction
        std::cout << "\n=== Running inference on the image... ===" << std::endl;
        int predictedClass = model.predict(image);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Predicted class: " << predictedClass << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== ERROR ===" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n=== UNKNOWN ERROR ===" << std::endl;
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}