#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Define a custom CNN model for object detection
struct SimpleCNNImpl : torch::nn::Module {
    SimpleCNNImpl() {
        // First convolutional layer: 3 input channels, 16 output channels, 3x3 kernel
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)));
        
        // First batch normalization layer
        batch_norm1 = register_module("batch_norm1", torch::nn::BatchNorm2d(16));
        
        // Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)));
        
        // Second batch normalization layer
        batch_norm2 = register_module("batch_norm2", torch::nn::BatchNorm2d(32));
        
        // Third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        
        // Third batch normalization layer
        batch_norm3 = register_module("batch_norm3", torch::nn::BatchNorm2d(64));
        
        // Fully connected layer for classification: 64*28*28 input features, 10 output classes
        // The 28x28 is the resulting feature map size after three 2x2 max pooling operations on 224x224 input
        fc1 = register_module("fc1", torch::nn::Linear(64 * 28 * 28, 512));
        
        // Output layer for bounding box regression: 512 input features, 4 outputs (x, y, width, height)
        bbox_regressor = register_module("bbox_regressor", torch::nn::Linear(512, 4));
        
        // Output layer for classification: 512 input features, num_classes outputs
        classifier = register_module("classifier", torch::nn::Linear(512, 20)); // Assuming 20 classes
    }

    // Define the forward pass
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // First convolutional block with ReLU activation and max pooling
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::max_pool2d(x, 2);
        
        // Second convolutional block with ReLU activation and max pooling
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::max_pool2d(x, 2);
        
        // Third convolutional block with ReLU activation and max pooling
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::max_pool2d(x, 2);
        
        // Flatten the tensor for the fully connected layer
        x = x.view({-1, 64 * 28 * 28});
        
        // Fully connected layer with ReLU activation
        x = torch::relu(fc1(x));
        
        // Output bounding box coordinates
        auto bbox_output = bbox_regressor(x);
        
        // Output class probabilities
        auto class_output = torch::log_softmax(classifier(x), 1);
        
        return {class_output, bbox_output};
    }

    // Declare the layers
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
    torch::nn::Linear fc1, bbox_regressor, classifier;
};

// Register the module
TORCH_MODULE(SimpleCNN);

// Data preprocessing utility
torch::Tensor preprocess_image(const cv::Mat& image, int input_size = 224) {
    cv::Mat processed_image;
    
    // Resize the image
    cv::resize(image, processed_image, cv::Size(input_size, input_size));
    
    // Convert to RGB
    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    processed_image.convertTo(processed_image, CV_32F, 1.0f / 255.0f);
    
    // Normalize using ImageNet mean and std
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(processed_image, mean, processed_image);
    cv::divide(processed_image, std, processed_image);
    
    // Convert OpenCV Mat to Tensor
    auto tensor_image = torch::from_blob(
        processed_image.data,
        {1, processed_image.rows, processed_image.cols, 3},
        torch::kFloat32
    );
    
    // Rearrange from NHWC to NCHW (batch, channels, height, width)
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    
    return tensor_image;
}

// Custom dataset class for object detection
class ObjectDetectionDataset : public torch::data::Dataset<ObjectDetectionDataset> {
private:
    std::vector<std::string> image_paths_;
    std::vector<std::vector<float>> bboxes_;
    std::vector<int> labels_;
    int input_size_;

public:
    ObjectDetectionDataset(
        const std::vector<std::string>& image_paths,
        const std::vector<std::vector<float>>& bboxes,
        const std::vector<int>& labels,
        int input_size = 224
    ) : image_paths_(image_paths), bboxes_(bboxes), labels_(labels), input_size_(input_size) {}

    // Return the size of the dataset
    torch::optional<size_t> size() const override {
        return image_paths_.size();
    }

    // Get item at the given index
    torch::data::Example<> get(size_t index) override {
        // Load the image
        cv::Mat image = cv::imread(image_paths_[index]);
        if (image.empty()) {
            throw std::runtime_error("Could not read image: " + image_paths_[index]);
        }
        
        // Preprocess the image
        torch::Tensor tensor_image = preprocess_image(image, input_size_);
        
        // Create a tensor from the bounding box
        auto bbox = torch::tensor(bboxes_[index], torch::kFloat32);
        
        // Create a tensor from the label
        auto label = torch::tensor(labels_[index], torch::kLong);
        
        // Return a pair of tensors
        return {tensor_image, torch::stack({label, bbox})};
    }
};

// Utility function to draw bounding boxes on images
void draw_bounding_box(cv::Mat& image, const std::vector<float>& bbox, const std::string& label) {
    int x = static_cast<int>(bbox[0] * image.cols);
    int y = static_cast<int>(bbox[1] * image.rows);
    int width = static_cast<int>(bbox[2] * image.cols);
    int height = static_cast<int>(bbox[3] * image.rows);
    
    // Draw rectangle
    cv::rectangle(image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);
    
    // Draw label
    cv::putText(
        image, 
        label, 
        cv::Point(x, y - 5), 
        cv::FONT_HERSHEY_SIMPLEX, 
        0.5, 
        cv::Scalar(0, 255, 0), 
        1
    );
}

// Main function
int main() {
    // Device setup (use CUDA if available)
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    // Hyperparameters
    const int64_t num_epochs = 10;
    const int64_t batch_size = 16;
    const double learning_rate = 0.001;
    
    // Create an instance of our model
    SimpleCNN model;
    model->to(device);
    
    // Define the optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    
    // Define loss functions
    auto classification_loss_fn = torch::nn::NLLLoss();
    auto regression_loss_fn = torch::nn::SmoothL1Loss();
    
    // Example of training loop (in reality, you would load a dataset)
    std::cout << "Starting training..." << std::endl;
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Set the model to training mode
        model->train();
        
        // Training code would go here...
        // For each batch:
        //   - Forward pass
        //   - Calculate loss
        //   - Backward pass
        //   - Optimizer step
        
        // After each epoch, validate the model
        model->eval();
        
        // Validation code would go here...
        
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "]" << std::endl;
    }
    
    // Once trained, save the model
    torch::save(model, "object_detection_model.pt");
    std::cout << "Model saved!" << std::endl;
    
    // Example of inference
    std::cout << "Running inference on test image..." << std::endl;
    
    // Load a test image
    cv::Mat test_image = cv::imread("test_image.jpg");
    if (test_image.empty()) {
        std::cerr << "Could not read the test image" << std::endl;
        return 1;
    }
    
    // Display the original image
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", test_image);
    
    // Preprocess the image
    torch::Tensor tensor_image = preprocess_image(test_image);
    tensor_image = tensor_image.to(device);
    
    // Run inference
    model->eval();
    torch::NoGradGuard no_grad;
    auto outputs = model->forward(tensor_image);
    
    auto class_output = outputs.first;
    auto bbox_output = outputs.second;
    
    // Get the predicted class
    auto class_probabilities = torch::exp(class_output);
    auto [max_prob, class_idx] = torch::max(class_probabilities, 1);
    
    // Convert bbox tensor to vector
    std::vector<float> bbox(4);
    auto bbox_accessor = bbox_output.accessor<float, 2>();
    for (int i = 0; i < 4; ++i) {
        bbox[i] = bbox_accessor[0][i];
    }
    
    // Map class index to class name (would load from file in real app)
    std::vector<std::string> class_names = {"person", "car", "dog", "cat", "bird", "..."}; // Example
    std::string class_name = class_names[class_idx.item<int>()];
    
    // Draw bounding box on the image
    draw_bounding_box(test_image, bbox, class_name);
    
    // Display the result
    cv::namedWindow("Detection Result", cv::WINDOW_NORMAL);
    cv::imshow("Detection Result", test_image);
    cv::waitKey(0);
    
    return 0;
}