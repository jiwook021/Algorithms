#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

// Include our custom wrapper instead of OpenCV directly
#include "opencv_wrapper.h"

// Include LibTorch headers
#include <torch/torch.h>
#include <torch/script.h>

// Helper class to manage Mat pointers
class MatPtr {
public:
    MatPtr() : ptr(nullptr) {}
    explicit MatPtr(cv::Mat* p) : ptr(p) {}
    ~MatPtr() { if (ptr) delete ptr; }
    
    // Disable copying
    MatPtr(const MatPtr&) = delete;
    MatPtr& operator=(const MatPtr&) = delete;
    
    // Move operations
    MatPtr(MatPtr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    MatPtr& operator=(MatPtr&& other) {
        if (this != &other) {
            if (ptr) delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Access
    cv::Mat* get() const { return ptr; }
    bool empty() const { return !ptr || ptr->empty(); }
    
private:
    cv::Mat* ptr;
};

class ImageClassifier {
private:
    torch::jit::script::Module model_;
    std::vector<std::string> class_names_;
    int input_size_;
    
public:
    // Constructor that loads a pre-trained model from a file
    ImageClassifier(const std::string& model_path, const std::string& classes_path, int input_size = 224) 
        : input_size_(input_size) {
        try {
            // Load the TorchScript model
            model_ = torch::jit::load(model_path);
            model_.eval(); // Set to evaluation mode
            
            // Load class names from file
            std::ifstream file(classes_path);
            std::string line;
            while (std::getline(file, line)) {
                class_names_.push_back(line);
            }
            
            std::cout << "Model loaded successfully with " << class_names_.size() << " classes." << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Preprocess the input image to match model requirements
    torch::Tensor preprocess(cv::Mat* input_image) {
        // Create a new Mat for processing
        cv::Mat* image = new cv::Mat();
        
        // Resize the image
        resize_image(input_image, image, cv::Size(input_size_, input_size_));
        
        // Convert from BGR to RGB
        cvt_color(image, image, COLOR_BGR2RGB);
        
        // Convert to float and normalize to [0, 1]
        // We can't directly use convertTo through our wrapper, so we handle this in LibTorch
        
        // Create a tensor from the Mat data
        auto tensor_image = torch::from_blob(
            image->data, 
            {1, image->rows, image->cols, 3}, 
            torch::kByte  // Use kByte for uint8_t data
        ).to(torch::kFloat32).div(255.0);  // Convert to float and normalize in LibTorch
        
        // Normalize using ImageNet mean and std
        tensor_image = tensor_image.sub(torch::tensor({0.485, 0.456, 0.406})).div(torch::tensor({0.229, 0.224, 0.225}));
        
        // Rearrange from NHWC to NCHW (batch, channels, height, width)
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        
        // Clean up the temporary Mat
        delete image;
        
        return tensor_image;
    }
    
    // Predict the class of the input image
    std::pair<std::string, float> predict(cv::Mat* image) {
        if (!image) {
            throw std::runtime_error("Null image provided");
        }
        
        // Preprocess the image
        torch::Tensor tensor_image = preprocess(image);
        
        // Create a vector of inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        
        // Execute the model and turn its output into a tensor
        torch::NoGradGuard no_grad; // Disable gradient calculation for inference
        auto output = model_.forward(inputs).toTensor();
        
        // Apply softmax to get probabilities
        auto probabilities = torch::softmax(output, 1);
        
        // Get the predicted class
        auto max_result = torch::max(probabilities, 1);
        auto values = std::get<0>(max_result);    // First tensor contains the maximum values
        auto indices = std::get<1>(max_result);   // Second tensor contains the indices
        int class_idx = indices.item<int>();
        float confidence = values.item<float>();
        
        // Return class name and confidence
        return std::make_pair(class_names_[class_idx], confidence);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    std::string model_path = "model/resnet18.pt";  // Path to your saved TorchScript model
    std::string classes_path = "model/imagenet_classes.txt";  // Path to class names file
    
    try {
        // Load the image using our wrapper
        MatPtr image(load_image(image_path));
        if (image.empty()) {
            std::cerr << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
        
        // Display the original image
        create_window("Original Image", WINDOW_NORMAL_CONST);
        show_image("Original Image", image.get());
        
        // Initialize the classifier
        ImageClassifier classifier(model_path, classes_path);
        
        // Make a prediction
        auto result = classifier.predict(image.get());
        std::string class_name = result.first;
        float confidence = result.second;
        
        // Print the results
        std::cout << "Prediction: " << class_name << std::endl;
        std::cout << "Confidence: " << confidence * 100.0f << "%" << std::endl;
        
        // Draw the prediction on the image
        draw_text(
            image.get(), 
            class_name + " (" + std::to_string(int(confidence * 100)) + "%)", 
            cv::Point(20, 40), 
            FONT_HERSHEY_SIMPLEX, 
            1.0, 
            cv::Scalar(0, 255, 0), 
            2
        );
        
        // Display the result
        create_window("Prediction", WINDOW_NORMAL_CONST);
        show_image("Prediction", image.get());
        wait_key(0);
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}