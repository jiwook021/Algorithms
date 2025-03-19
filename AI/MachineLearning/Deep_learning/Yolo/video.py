from ultralytics import YOLO
import cv2
import time
import os

def detect_video_file(video_path, output_path=None, confidence_threshold=0.25):
    """
    Detect objects in a video file using YOLOv8.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output video (None to not save)
        confidence_threshold (float): Minimum confidence threshold for detections
        
    Returns:
        None
    """
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Load the YOLOv8 model
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")  # Load the smallest YOLOv8 model
        
        # Initialize video capture
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} total frames")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            print(f"Output will be saved to: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frames
        frame_count = 0
        processing_times = []
        
        print("Processing video... Press 'q' to quit.")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Record start time
            start_time = time.time()
            
            # Perform detection
            results = model(frame, conf=confidence_threshold)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Get processed frame with bounding boxes
            processed_frame = results[0].plot()
            
            # Display processing info on frame
            avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Show FPS and progress
            cv2.putText(processed_frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            progress = (frame_count / total_frames) * 100
            cv2.putText(processed_frame, f"Progress: {progress:.1f}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display detection counts
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes.cpu().numpy()
                num_detections = len(boxes)
                cv2.putText(processed_frame, f"Detections: {num_detections}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the processed frame
            cv2.imshow("YOLO Video Detection", processed_frame)
            
            # Write frame to output video if specified
            if writer:
                writer.write(processed_frame)
            
            # Increment frame counter
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection stopped by user")
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_processing_time
            print(f"\nPerformance Statistics:")
            print(f"  Total frames processed: {frame_count}/{total_frames}")
            print(f"  Average processing time: {avg_processing_time:.4f} seconds per frame")
            print(f"  Average FPS: {avg_fps:.2f}")
            
            if output_path:
                print(f"Output video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error in video detection: {str(e)}")
        # Clean up resources
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'writer' in locals() and writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    # Example usage
    video_path = "sample/traffic_bus.mp4"  # Replace with your video path
    output_path = "samples/output_video.mp4"       # Replace with desired output path
    
    # Run detection
    detect_video_file(
        video_path=video_path,
        output_path=output_path,
        confidence_threshold=0.25
    )