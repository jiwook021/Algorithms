# Code Overview: Viewer.cpp

### Purpose of the Code

The provided code is part of a C++ program that implements a `Viewer` class, which is designed to visualize and display the results of object detection and tracking in real-time video processing. This is commonly used in computer vision applications, such as surveillance systems, autonomous vehicles, or any system that requires real-time object recognition and tracking.

### Main Functionality

1. **Window Creation and Management**:
   - The `Viewer` class creates a window using OpenCV (`cv::namedWindow`) where the video frames will be displayed. This window can be resized and manipulated by the user.

2. **Displaying Video Frames**:
   - The `display` method is responsible for showing the processed video frames. It takes a frame (image) as input, along with detected objects and tracked objects, and displays them in the window.

3. **Drawing Detections and Tracks**:
   - The code includes functionality to draw bounding boxes around detected objects (`drawDetections`) and to visualize the tracks of these objects over time (`drawTracks`). This helps in visually understanding where objects are and how they move.

4. **Performance Metrics**:
   - The code also supports displaying performance metrics such as inference time (time taken to detect objects) and processing time (time taken to process the frame). This is useful for optimizing and debugging the system.

5. **Error Handling**:
   - The code includes basic error handling to ensure that invalid frames (empty frames) are not processed, and any exceptions during the display process are caught and logged.

### Algorithms Used

- **Object Detection**: Although not explicitly shown in the provided code, the `Detection` objects likely come from an object detection algorithm (e.g., YOLO, SSD, or Faster R-CNN). These algorithms identify objects in the frame and return their bounding boxes and class labels.

- **Object Tracking**: The `TrackedObject` objects suggest that a tracking algorithm (e.g., SORT, DeepSORT, or Kalman Filter-based tracking) is used to follow objects across frames. This helps in maintaining the identity of objects as they move.

- **OpenCV Functions**: The code heavily relies on OpenCV functions for image processing (`cv::Mat`, `cv::namedWindow`, `cv::imshow`, etc.). OpenCV is a powerful library for real-time computer vision.

### Overall Structure

1. **Viewer Class**:
   - The `Viewer` class encapsulates all the functionality related to displaying video frames, drawing detections and tracks, and showing performance metrics. It is initialized with a window name, class names (for detected objects), and a flag to control whether to show FPS (frames per second).

2. **Constructor**:
   - The constructor (`Viewer::Viewer`) initializes the class members and creates the display window.

3. **Display Method**:
   - The `display` method is the core function that processes each frame. It checks if the frame is valid, clones it for drawing, and then draws detections and tracks if enabled. Finally, it displays the frame in the window and waits for a key press.

4. **Error Handling**:
   - The `try-catch` block in the `display` method ensures that any exceptions during the display process are caught and logged, preventing the program from crashing.

### Problem Being Solved

The code addresses the problem of visualizing the results of real-time object detection and tracking in video streams. This is crucial for applications where understanding the spatial and temporal dynamics of objects is important, such as in surveillance, autonomous driving, or human-computer interaction.

### Approach Taken

- **Modular Design**: The `Viewer` class is designed to be modular, allowing it to be easily integrated into larger systems. It separates the concerns of visualization from the actual detection and tracking logic.

- **Real-Time Display**: The code is designed to handle real-time video streams, ensuring that frames are processed and displayed with minimal delay.

- **Customizability**: The class allows for customization of what is displayed (detections, tracks, FPS) through member variables and flags.

### How Different Parts Work Together

- **Initialization**: When a `Viewer` object is created, it sets up the display window and initializes the necessary parameters.

- **Frame Processing**: For each frame, the `display` method is called. This method checks the frame's validity, clones it, and then draws the necessary information (detections, tracks) on the cloned frame.

- **Display**: The processed frame is then displayed in the window. The `cv::waitKey` function ensures that the frame is displayed for a sufficient amount of time and allows for user interaction (e.g., pressing a key to exit).

- **Error Handling**: If anything goes wrong during the display process, the exception is caught, and an error message is printed, ensuring that the program can continue running or exit gracefully.

In summary, this code provides a robust and flexible way to visualize the results of object detection and tracking in real-time video streams, making it easier to understand and debug computer vision systems.