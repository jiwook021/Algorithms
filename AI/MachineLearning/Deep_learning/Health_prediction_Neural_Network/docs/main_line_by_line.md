# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the design choices.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <iomanip>
```

#### **What It Does**
These lines include libraries that provide functionality for:
- Input/output (`iostream`, `fstream`, `iomanip`)
- Data structures (`vector`, `string`, `unordered_map`)
- Random number generation (`random`)
- Mathematical operations (`cmath`, `algorithm`)
- Thread safety (`mutex`)
- Memory management (`memory`)

#### **Why It’s Used**
- **`<vector>`**: Used to store dynamic arrays (e.g., layers of neurons).
- **`<random>`**: Used to initialize weights randomly.
- **`<mutex>`**: Ensures thread safety during inference.
- **`<cmath>`**: Provides mathematical functions like `exp`, `tanh`, and `sqrt`.

---

### **2. Class Definition and Activation Types**
```cpp
class HealthScorePredictor {
public:
    enum class ActivationType {
        SIGMOID,
        RELU,
        TANH,
        LINEAR
    };
```

#### **What It Does**
- Defines a class `HealthScorePredictor` to encapsulate the neural network.
- Declares an `enum class` for activation function types.

#### **Why It’s Used**
- **Activation Functions**: These functions determine how a neuron’s output is calculated based on its input. Different functions are used for different purposes:
  - **Sigmoid**: Smooth curve, useful for binary classification.
  - **ReLU**: Simple and efficient, commonly used in hidden layers.
  - **Tanh**: Similar to Sigmoid but outputs values in the range [-1, 1].
  - **Linear**: No transformation, used for regression tasks.

---

### **3. Feature Struct**
```cpp
struct Feature {
    std::string name;
    double min_value;
    double max_value;
    double mean;
    double std_dev;

    double normalize(double value) const {
        if (std_dev > 0) {
            return (value - mean) / std_dev; // Z-score normalization
        } else {
            return (value - min_value) / (max_value - min_value) * 2.0 - 1.0; // Min-max scaling
        }
    }

    double denormalize(double normalized_value) const {
        if (std_dev > 0) {
            return normalized_value * std_dev + mean; // Z-score denormalization
        } else {
            return (normalized_value + 1.0) / 2.0 * (max_value - min_value) + min_value; // Min-max denormalization
        }
    }
};
```

#### **What It Does**
- Represents a feature (e.g., blood pressure, cholesterol) with its name, min/max values, mean, and standard deviation.
- Provides methods to normalize and denormalize values.

#### **Why It’s Used**
- **Normalization**: Ensures all features are on a similar scale, which helps the neural network learn more efficiently.
  - **Z-score normalization**: Scales values based on mean and standard deviation.
  - **Min-max scaling**: Scales values to a range of [-1, 1].
- **Denormalization**: Converts normalized values back to their original scale for interpretation.

#### **Example**
If a feature has:
- `min_value = 50`, `max_value = 150`, `mean = 100`, `std_dev = 20`
- A value of `120` would be normalized to:
  - Z-score: `(120 - 100) / 20 = 1.0`
  - Min-max: `(120 - 50) / (150 - 50) * 2.0 - 1.0 = 0.4`

---

### **4. Neuron and Layer Structs**
```cpp
struct Connection {
    double weight;
    double delta_weight;
};

struct Neuron {
    std::vector<Connection> output_weights;
    double output_value;
    double gradient;
    unsigned index;

    Neuron(unsigned num_outputs, unsigned idx) : output_value(0.0), gradient(0.0), index(idx) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<double> dist(0.0, 1.0);

        output_weights.resize(num_outputs);
        for (auto& connection : output_weights) {
            connection.weight = dist(gen) * sqrt(2.0 / (num_outputs + 1)); // Xavier/Glorot initialization
        }
    }
};

struct Layer {
    std::vector<Neuron> neurons;
    ActivationType activation;

    Layer(unsigned num_neurons, unsigned num_outputs, ActivationType act) : activation(act) {
        neurons.reserve(num_neurons + 1); // +1 for bias neuron
        for (unsigned i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_outputs, i);
        }
        neurons.emplace_back(num_outputs, num_neurons); // Bias neuron
        neurons.back().output_value = 1.0;
    }
};
```

#### **What It Does**
- **Connection**: Represents a connection between neurons, with a weight and a delta weight (used during training).
- **Neuron**: Represents a single neuron, with:
  - `output_weights`: Connections to neurons in the next layer.
  - `output_value`: The neuron’s output after applying the activation function.
  - `gradient`: Used during backpropagation to adjust weights.
  - `index`: The neuron’s position in the layer.
- **Layer**: Represents a layer of neurons, with:
  - `neurons`: A vector of neurons.
  - `activation`: The activation function for the layer.

#### **Why It’s Used**
- **Xavier/Glorot Initialization**: Initializes weights to prevent vanishing or exploding gradients.
- **Bias Neuron**: Adds a constant value (1.0) to help the network learn better.

#### **Example**
For a layer with 3 neurons and 2 outputs:
- Each neuron will have 2 connections (one for each output).
- The bias neuron ensures the network can learn even when all inputs are zero.

---

### **5. Activation Functions**
```cpp
double activate(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ActivationType::RELU:
            return std::max(0.0, x);
        case ActivationType::TANH:
            return tanh(x);
        case ActivationType::LINEAR:
        default:
            return x;
    }
}

double activate_derivative(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::SIGMOID:
            return x * (1.0 - x);
        case ActivationType::RELU:
            return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::TANH:
            return 1.0 - x * x;
        case ActivationType::LINEAR:
        default:
            return 1.0;
    }
}
```

#### **What It Does**
- **`activate`**: Applies the activation function to a neuron’s input.
- **`activate_derivative`**: Computes the derivative of the activation function, used during backpropagation.

#### **Why It’s Used**
- **Activation Functions**: Introduce non-linearity, allowing the network to learn complex patterns.
- **Derivatives**: Used to calculate gradients during backpropagation.

#### **Example**
For a ReLU activation:
- Input: `x = -1.0` → Output: `0.0`
- Input: `x = 2.0` → Output: `2.0`
- Derivative: `1.0` if `x > 0`, else `0.0`

---

### **6. Feedforward Propagation**
```cpp
void feed_forward(const std::vector<double>& input_values) {
    for (unsigned i = 0; i < input_values.size(); ++i) {
        layers[0].neurons[i].output_value = input_values[i];
    }

    for (unsigned layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
        auto& prev_layer = layers[layer_idx - 1];
        auto& current_layer = layers[layer_idx];

        for (unsigned n = 0; n < current_layer.neurons.size() - 1; ++n) {
            double sum = 0.0;
            for (unsigned prev_n = 0; prev_n < prev_layer.neurons.size(); ++prev_n) {
                sum += prev_layer.neurons[prev_n].output_value * 
                       prev_layer.neurons[prev_n].output_weights[n].weight;
            }
            current_layer.neurons[n].output_value = activate(sum, current_layer.activation);
        }
    }
}
```

#### **What It Does**
- Passes input values through the network to compute the output.
- For each layer, calculates the weighted sum of inputs and applies the activation function.

#### **Why It’s Used**
- **Feedforward Propagation**: Computes predictions based on input data.

#### **Example**
For a network with 2 layers:
1. Input layer: Sets input values.
2. Hidden layer: Computes weighted sums and applies activation functions.

---

This is just the beginning! Let me know if you’d like me to continue with the rest of the code.