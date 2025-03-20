# Code Overview: main.cu

This CUDA code performs **2D convolution** on an image using a GPU for acceleration. Let's break down its purpose, functionality, and structure in detail:

---

### **Purpose of the Code**
The code is designed to apply a **2D convolution operation** to an image using a GPU. Convolution is a fundamental operation in image processing, often used for tasks like edge detection, blurring, sharpening, and feature extraction. The GPU is used to accelerate this computationally intensive task by parallelizing the convolution operation across thousands of threads.

The specific problem being solved is **edge detection** using a **Sobel kernel**, which is a common technique for detecting horizontal or vertical edges in images. The code takes an input image, applies the convolution operation using the Sobel kernel, and produces an output image highlighting the detected edges.

---

### **Main Functionality**
1. **Image Loading and Preprocessing**:
   - The input image is loaded as a grayscale image and converted to a floating-point format (normalized to the range [0, 1]).
   - This preprocessing step ensures the image is in a format suitable for convolution.

2. **Convolution Operation**:
   - A 2D convolution is performed on the image using a predefined kernel (in this case, a Sobel kernel for horizontal edge detection).
   - The convolution operation is parallelized using CUDA, where each thread computes a single pixel of the output image.

3. **Postprocessing**:
   - The output image is normalized to the range [0, 255] and converted back to an 8-bit format for saving.
   - The final image is saved as `output.png`.

---

### **Algorithms Used**
1. **2D Convolution**:
   - Convolution is a mathematical operation where a kernel (a small matrix) is slid over an image, and at each position, the sum of element-wise multiplications between the kernel and the overlapping region of the image is computed.
   - The formula for 2D convolution is:
     \[
     \text{output}(x, y) = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}(x+i, y+j) \cdot \text{kernel}(i, j)
     \]
     where \(k_h\) and \(k_w\) are the height and width of the kernel, respectively.

2. **CUDA Parallelization**:
   - The convolution operation is parallelized using CUDA, where each thread computes the convolution result for a single pixel in the output image.
   - The GPU's massive parallelism allows for significant speedup compared to a CPU implementation.

---

### **Overall Structure**
The code is structured into three main parts:

1. **CUDA Kernel (`conv2D`)**:
   - This is the core computation executed on the GPU.
   - Each thread computes the convolution result for one pixel in the output image.
   - The kernel uses nested loops to iterate over the kernel dimensions and compute the weighted sum.

2. **Host-Side Helper Function (`runConvolution`)**:
   - Manages memory allocation, data transfer between the CPU and GPU, and kernel invocation.
   - Computes the grid and block dimensions for the CUDA kernel.
   - Handles the input and output image data.

3. **Main Function**:
   - Loads the input image and preprocesses it.
   - Defines the Sobel kernel for edge detection.
   - Calls the `runConvolution` function to perform the convolution.
   - Postprocesses and saves the output image.

---

### **How the Parts Work Together**
1. **Input Image**:
   - The input image is loaded and preprocessed in the `main` function.

2. **Kernel Definition**:
   - The Sobel kernel is defined as a vector of floats in the `main` function.

3. **Convolution Execution**:
   - The `runConvolution` function handles the GPU-side operations:
     - Allocates memory on the GPU for the input image, kernel, and output image.
     - Copies the input image and kernel to the GPU.
     - Launches the CUDA kernel (`conv2D`) with appropriate grid and block dimensions.
     - Copies the result back to the CPU.

4. **Output Image**:
   - The output image is normalized and saved in the `main` function.

---

### **Key Components**
1. **CUDA Error Checking**:
   - The `CUDA_CHECK` macro ensures that CUDA API calls are successful, providing detailed error messages if something goes wrong.

2. **Grid and Block Dimensions**:
   - The grid and block dimensions are calculated to ensure all pixels in the output image are processed.
   - Each thread computes one pixel, and the grid is sized to cover the entire output image.

3. **Memory Management**:
   - Memory is allocated on the GPU for the input image, kernel, and output image.
   - Data is transferred between the CPU and GPU using `cudaMemcpy`.

4. **Sobel Kernel**:
   - The Sobel kernel is a 3x3 matrix designed to detect horizontal edges in the image.

---

### **Summary**
This code demonstrates how to use CUDA to accelerate a 2D convolution operation for image processing. It loads an image, applies a Sobel kernel for edge detection, and saves the result. The GPU's parallelism is leveraged to perform the convolution efficiently, making it much faster than a CPU implementation for large images. The code is structured to handle memory management, error checking, and data transfer between the CPU and GPU, ensuring robust and efficient execution.