#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <cstdint>  // For uint32_t
#include <iomanip>  // For std::setw and std::setfill

// Structure for MNIST dataset
struct MNISTDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

// Simple neural network structure
struct NeuralNetwork {
    std::vector<float> weights;
    float bias;
    int inputSize;

    NeuralNetwork(int size) : inputSize(size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);
        weights.resize(inputSize);
        for (int i = 0; i < inputSize; ++i) {
            weights[i] = dis(gen);
        }
        bias = dis(gen);
    }

    float forward(const std::vector<float>& input) const {
        float sum = bias;
        for (int i = 0; i < inputSize; ++i) {
            sum += weights[i] * input[i];
        }
        return sigmoid(sum);
    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

// Function to read MNIST dataset files (in IDX format)
MNISTDataset readMNISTDataset(const std::string& imagesFile, const std::string& labelsFile) {
    MNISTDataset dataset;

    // Open images file
    std::ifstream imagesStream(imagesFile, std::ios::binary);
    if (!imagesStream) {
        std::cerr << "Error opening images file: " << imagesFile << std::endl;
        return dataset;
    }

    // Debug: Print first 16 bytes of images file
    unsigned char imgBuffer[16];
    imagesStream.read(reinterpret_cast<char*>(imgBuffer), 16);
    if (imagesStream.gcount() != 16) {
        std::cerr << "Failed to read 16 bytes from images file" << std::endl;
        return dataset;
    }
    std::cout << "First 16 bytes of images file: ";
    for (int i = 0; i < 16; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(imgBuffer[i]) << " ";
    }
    std::cout << std::dec << std::endl;
    imagesStream.seekg(0, std::ios::beg);  // Reset to beginning

    // Open labels file
    std::ifstream labelsStream(labelsFile, std::ios::binary);
    if (!labelsStream) {
        std::cerr << "Error opening labels file: " << labelsFile << std::endl;
        return dataset;
    }

    // Debug: Print first 8 bytes of labels file
    unsigned char lblBuffer[8];
    labelsStream.read(reinterpret_cast<char*>(lblBuffer), 8);
    if (labelsStream.gcount() != 8) {
        std::cerr << "Failed to read 8 bytes from labels file" << std::endl;
        return dataset;
    }
    std::cout << "First 8 bytes of labels file: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(lblBuffer[i]) << " ";
    }
    std::cout << std::dec << std::endl;
    labelsStream.seekg(0, std::ios::beg);  // Reset to beginning

    // Read images header
    uint32_t magicNumber, numImages, numRows, numCols;
    imagesStream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    if (imagesStream.gcount() != sizeof(magicNumber)) {
        std::cerr << "Failed to read magic number from images file" << std::endl;
        return dataset;
    }
    magicNumber = ((magicNumber & 0xff) << 24) | ((magicNumber & 0xff00) << 8) |
                  ((magicNumber & 0xff0000) >> 8) | ((magicNumber & 0xff000000) >> 24);

    imagesStream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    if (imagesStream.gcount() != sizeof(numImages)) {
        std::cerr << "Failed to read number of images from images file" << std::endl;
        return dataset;
    }
    numImages = ((numImages & 0xff) << 24) | ((numImages & 0xff00) << 8) |
                ((numImages & 0xff0000) >> 8) | ((numImages & 0xff000000) >> 24);

    imagesStream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    if (imagesStream.gcount() != sizeof(numRows)) {
        std::cerr << "Failed to read number of rows from images file" << std::endl;
        return dataset;
    }
    numRows = ((numRows & 0xff) << 24) | ((numRows & 0xff00) << 8) |
              ((numRows & 0xff0000) >> 8) | ((numRows & 0xff000000) >> 24);

    imagesStream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    if (imagesStream.gcount() != sizeof(numCols)) {
        std::cerr << "Failed to read number of columns from images file" << std::endl;
        return dataset;
    }
    numCols = ((numCols & 0xff) << 24) | ((numCols & 0xff00) << 8) |
              ((numCols & 0xff0000) >> 8) | ((numCols & 0xff000000) >> 24);

    // Read labels header
    uint32_t labelsMagicNumber, numLabels;
    labelsStream.read(reinterpret_cast<char*>(&labelsMagicNumber), sizeof(labelsMagicNumber));
    if (labelsStream.gcount() != sizeof(labelsMagicNumber)) {
        std::cerr << "Failed to read magic number from labels file" << std::endl;
        return dataset;
    }
    labelsMagicNumber = ((labelsMagicNumber & 0xff) << 24) | ((labelsMagicNumber & 0xff00) << 8) |
                        ((labelsMagicNumber & 0xff0000) >> 8) | ((labelsMagicNumber & 0xff000000) >> 24);

    labelsStream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    if (labelsStream.gcount() != sizeof(numLabels)) {
        std::cerr << "Failed to read number of labels from labels file" << std::endl;
        return dataset;
    }
    numLabels = ((numLabels & 0xff) << 24) | ((numLabels & 0xff00) << 8) |
                ((numLabels & 0xff0000) >> 8) | ((numLabels & 0xff000000) >> 24);

    // Verify headers
    if (numImages != numLabels) {
        std::cerr << "Error: Number of images (" << numImages 
                  << ") and labels (" << numLabels << ") don't match!" << std::endl;
        return dataset;
    }
    if (magicNumber != 2051 || labelsMagicNumber != 2049) {
        std::cerr << "Error: Invalid magic numbers - Images: " << magicNumber 
                  << ", Labels: " << labelsMagicNumber << std::endl;
        return dataset;
    }

    // Allocate memory
    const int imageSize = numRows * numCols;
    dataset.images.resize(numImages, std::vector<float>(imageSize));
    dataset.labels.resize(numImages);

    // Read images and labels
    for (uint32_t i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            unsigned char pixel;
            imagesStream.read(reinterpret_cast<char*>(&pixel), 1);
            if (imagesStream.gcount() != 1) {
                std::cerr << "Failed to read pixel " << j << " of image " << i << std::endl;
                return dataset;
            }
            dataset.images[i][j] = static_cast<float>(pixel) / 255.0f;
        }

        unsigned char label;
        labelsStream.read(reinterpret_cast<char*>(&label), 1);
        if (labelsStream.gcount() != 1) {
            std::cerr << "Failed to read label " << i << std::endl;
            return dataset;
        }
        dataset.labels[i] = static_cast<int>(label);
    }

    std::cout << "Successfully read " << numImages << " images and labels." << std::endl;
    return dataset;
}

// Function to train the neural network (simplified for one-vs-rest)
void trainNetwork(NeuralNetwork& nn, const MNISTDataset& trainData, int targetDigit, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0f;
        for (size_t i = 0; i < trainData.images.size(); ++i) {
            float target = (trainData.labels[i] == targetDigit) ? 1.0f : 0.0f;
            float output = nn.forward(trainData.images[i]);
            float error = target - output;

            // Update weights and bias
            for (int j = 0; j < nn.inputSize; ++j) {
                nn.weights[j] += learningRate * error * trainData.images[i][j];
            }
            nn.bias += learningRate * error;

            totalError += error * error;
        }
        std::cout << "Epoch " << epoch + 1 << ", Error: " << totalError << std::endl;
    }
}

// Function to evaluate the neural network
float evaluateNetwork(const NeuralNetwork& nn, const MNISTDataset& testData, int targetDigit) {
    int correct = 0;
    for (size_t i = 0; i < testData.images.size(); ++i) {
        float output = nn.forward(testData.images[i]);
        int prediction = (output >= 0.5f) ? targetDigit : -1;
        int actual = (testData.labels[i] == targetDigit) ? targetDigit : -1;
        if (prediction == actual) correct++;
    }
    return static_cast<float>(correct) / testData.images.size();
}

int main() {
    // File paths (assumed to exist and not zipped)
    std::string trainImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images-idx3-ubyte";
    std::string trainLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels-idx1-ubyte";
    std::string testImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images-idx3-ubyte";
    std::string testLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels-idx1-ubyte";

    // Read training and test datasets
    std::cout << "Reading training dataset..." << std::endl;
    MNISTDataset trainData = readMNISTDataset(trainImagesFile, trainLabelsFile);
    if (trainData.images.empty()) {
        std::cerr << "Failed to load training data." << std::endl;
        return 1;
    }

    std::cout << "Reading test dataset..." << std::endl;
    MNISTDataset testData = readMNISTDataset(testImagesFile, testLabelsFile);
    if (testData.images.empty()) {
        std::cerr << "Failed to load test data." << std::endl;
        return 1;
    }

    // Initialize neural networks (one per digit for simplicity)
    const int inputSize = 28 * 28;  // MNIST image size
    std::vector<NeuralNetwork> classifiers(10, NeuralNetwork(inputSize));

    // Train classifiers (one-vs-rest)
    std::cout << "Training classifiers..." << std::endl;
    for (int digit = 0; digit < 10; ++digit) {
        std::cout << "Training for digit " << digit << std::endl;
        trainNetwork(classifiers[digit], trainData, digit, 5, 0.01f);
    }

    // Evaluate classifiers
    std::cout << "Evaluating classifiers..." << std::endl;
    float totalAccuracy = 0.0f;
    for (int digit = 0; digit < 10; ++digit) {
        float accuracy = evaluateNetwork(classifiers[digit], testData, digit);
        std::cout << "Accuracy for digit " << digit << ": " << accuracy * 100 << "%" << std::endl;
        totalAccuracy += accuracy;
    }
    std::cout << "Average accuracy: " << (totalAccuracy / 10.0f) * 100 << "%" << std::endl;

    return 0;
}