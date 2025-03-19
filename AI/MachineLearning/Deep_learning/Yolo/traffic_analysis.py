from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import math
from collections import defaultdict, deque

# ===================== CONFIG =====================
# Classes of interest for different features
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
PERSON_CLASSES = ['person']
TRAFFIC_SIGN_CLASSES = ['traffic light', 'stop sign']
ALL_CLASSES = VEHICLE_CLASSES + PERSON_CLASSES + TRAFFIC_SIGN_CLASSES

# Feature flags - enable/disable as needed
ENABLE_VEHICLE_COUNTING = True
ENABLE_TRAFFIC_FLOW = True
ENABLE_TRAFFIC_SIGN = True
ENABLE_LANE_DETECTION = False
ENABLE_HUMAN_DETECTION = True  # Disabled as requested

# Thresholds and parameters
CONFIDENCE_THRESHOLD = 0.25
MAX_TRACKING_AGE = 5  # Reduced from 30 to 5 to make objects disappear faster
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for detection to be tracked
CONGESTION_THRESHOLD = 5  # Number of vehicles in ROI to consider congestion
# ====================================================

class TrafficAnalysisSystem:
    def __init__(self, 
                 model_path="yolov8n.pt", 
                 enable_vehicle_counting=True,
                 enable_traffic_flow=True,
                 enable_traffic_sign=True,
                 enable_lane_detection=True):
        """
        Initialize the traffic analysis system with YOLO.
        
        Args:
            model_path: Path to the YOLO model weights
            enable_*: Feature flags to enable/disable specific functionalities
        """
        print(f"Initializing Traffic Analysis System...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print(f"YOLO model loaded: {model_path}")
        
        # Enable/disable features
        self.enable_vehicle_counting = enable_vehicle_counting
        self.enable_traffic_flow = enable_traffic_flow
        self.enable_traffic_sign = enable_traffic_sign
        self.enable_lane_detection = enable_lane_detection
        
        # Initialize tracking system
        self.tracks = {}  # id -> track data
        self.next_id = 0  # next available tracking ID
        
        # Initialize counters
        self.vehicle_counts = {vehicle_class: 0 for vehicle_class in VEHICLE_CLASSES}
        self.total_vehicles = 0
        
        # Traffic flow analysis
        self.avg_speeds = deque(maxlen=100)  # store recent average speeds
        self.congestion_level = 0  # 0: none, 1: light, 2: moderate, 3: heavy
        self.vehicles_in_roi = []  # vehicles in region of interest
        
        # Traffic sign detection
        self.detected_signs = []
        self.traffic_light_states = {}  # track_id -> color
        
        # Lane detection parameters
        self.last_lanes = None
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        self.fps = 0  # For calculating velocity in terms of changes per second
        self.last_time = time.time()
        
        # No counting lines as requested
        self.counting_lines = []
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        
        # Flags for one-time initialization
        self.initialized = False
        
        print("System initialized and ready for video processing")
    
    def iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        # Box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection and union
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def is_box_in_frame(self, box):
        """Check if a bounding box is fully inside the frame"""
        x1, y1, x2, y2 = box
        
        # Check if box is outside frame boundaries
        if x2 < 0 or y2 < 0 or x1 > self.frame_width or y1 > self.frame_height:
            return False
            
        # Check if box is unreasonably large (which can happen with tracking errors)
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0 or width > self.frame_width*0.9 or height > self.frame_height*0.9:
            return False
            
        return True
    
    def update_tracks(self, detections, frame):
        """Update tracking information with new detections using a greedy approach"""
        # Increment age of all current tracks
        for track_id in self.tracks:
            self.tracks[track_id]['age'] += 1
        
        # Extract current track boxes and active tracks (not too old)
        active_tracks = {}
        for track_id, track_data in self.tracks.items():
            if track_data['age'] < MAX_TRACKING_AGE:
                active_tracks[track_id] = track_data
        
        # Create a list of unmatched detections
        unmatched_detections = []
        for det_idx, det in enumerate(detections):
            if det['confidence'] >= MIN_DETECTION_CONFIDENCE:
                unmatched_detections.append((det_idx, det))
        
        # Mark all detections as not matched yet
        for det in detections:
            det['matched'] = False
        
        # Greedy matching: For each track, find the best matching detection
        for track_id, track_data in active_tracks.items():
            best_iou = 0.3  # Minimum IoU threshold for a match
            best_det_idx = -1
            
            # Find the best detection for this track
            for i, (det_idx, det) in enumerate(unmatched_detections):
                if not det['matched']:  # Only consider unmatched detections
                    current_iou = self.iou(track_data['bbox'], det['bbox'])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_det_idx = i
            
            # If a match was found, update the track
            if best_det_idx >= 0:
                det_idx, det = unmatched_detections[best_det_idx]
                
                # Update track with new detection
                old_pos = None
                if len(track_data['positions']) > 0:
                    old_pos = track_data['positions'][-1]
                
                # Calculate center position
                center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                new_pos = (int(center_x), int(center_y))
                
                # Update track data
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['class'] = det['class']
                self.tracks[track_id]['confidence'] = det['confidence']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['positions'].append(new_pos)
                
                # Calculate velocity vector (pixels per second)
                if old_pos is not None and self.fps > 0:
                    dx = new_pos[0] - old_pos[0]
                    dy = new_pos[1] - old_pos[1]
                    velocity_magnitude = math.sqrt(dx*dx + dy*dy) * self.fps
                    velocity_angle = math.atan2(dy, dx)
                    self.tracks[track_id]['velocity'] = (velocity_magnitude, velocity_angle)
                    
                    # Calculate acceleration (change in velocity)
                    if 'velocity_history' in self.tracks[track_id] and self.tracks[track_id]['velocity_history']:
                        old_vel = self.tracks[track_id]['velocity_history'][-1]
                        dv = velocity_magnitude - old_vel[0]
                        self.tracks[track_id]['acceleration'] = dv * self.fps  # acceleration in pixels per second²
                    
                    # Store velocity history
                    if 'velocity_history' not in self.tracks[track_id]:
                        self.tracks[track_id]['velocity_history'] = []
                    self.tracks[track_id]['velocity_history'].append((velocity_magnitude, velocity_angle))
                    
                    # Keep history manageable
                    if len(self.tracks[track_id]['velocity_history']) > 10:
                        self.tracks[track_id]['velocity_history'].pop(0)
                
                # Clean positions list if too long
                if len(self.tracks[track_id]['positions']) > 30:
                    self.tracks[track_id]['positions'].pop(0)
                
                # Check for traffic light color if this is a traffic light
                if det['class'] == 'traffic light':
                    self._detect_traffic_light_color(frame, det['bbox'], track_id)
                
                # Mark this detection as matched
                det['matched'] = True
                
                # Remove this detection from unmatched list
                unmatched_detections.pop(best_det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in unmatched_detections:
            if det['confidence'] >= MIN_DETECTION_CONFIDENCE:
                # Initialize new track
                center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                
                new_track = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'counted': False,  # For counting
                    'positions': [(int(center_x), int(center_y))],
                    'first_seen': self.frame_count,
                    'velocity': None,
                    'velocity_history': [],
                    'acceleration': None
                }
                
                # Check for traffic light color if this is a traffic light
                if det['class'] == 'traffic light':
                    self._detect_traffic_light_color(frame, det['bbox'], self.next_id)
                
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Remove old tracks and tracks outside the frame
        tracks_to_keep = {}
        for track_id, track_data in self.tracks.items():
            if track_data['age'] < MAX_TRACKING_AGE and self.is_box_in_frame(track_data['bbox']):
                tracks_to_keep[track_id] = track_data
        self.tracks = tracks_to_keep
        
        # Calculate traffic flow metrics
        if self.enable_traffic_flow:
            self._analyze_traffic_flow(frame)
    
    def _detect_traffic_light_color(self, frame, bbox, track_id):
        """Detect the color of a traffic light by analyzing the ROI"""
        try:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            # Make sure the box is inside the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.frame_width, x2)
            y2 = min(self.frame_height, y2)
            
            # Extract the traffic light region
            light_roi = frame[y1:y2, x1:x2]
            if light_roi.size == 0:
                return
            
            # Convert to HSV color space for better color detection
            hsv_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for traffic lights
            # Red has two ranges in HSV (wraps around 180)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([90, 255, 255])
            
            # Create masks for each color
            mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
            
            # Count pixels of each color
            red_pixels = cv2.countNonZero(mask_red)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            green_pixels = cv2.countNonZero(mask_green)
            
            # Determine the dominant color
            total_colored_pixels = red_pixels + yellow_pixels + green_pixels
            
            # Set a minimum threshold to avoid false detections
            min_pixels = 10
            
            if total_colored_pixels > min_pixels:
                if red_pixels > yellow_pixels and red_pixels > green_pixels:
                    self.traffic_light_states[track_id] = "RED"
                elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                    self.traffic_light_states[track_id] = "YELLOW"
                elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                    self.traffic_light_states[track_id] = "GREEN"
                else:
                    self.traffic_light_states[track_id] = "UNKNOWN"
            else:
                self.traffic_light_states[track_id] = "UNKNOWN"
                
        except Exception as e:
            print(f"Error detecting traffic light color: {str(e)}")
            self.traffic_light_states[track_id] = "UNKNOWN"
    
    def _analyze_traffic_flow(self, frame):
        """Analyze traffic flow and detect congestion"""
        # Update vehicles in ROI (use whole frame as ROI for simplicity)
        self.vehicles_in_roi = []
        
        # Calculate speeds and find vehicles in ROI
        for track_id, track_data in self.tracks.items():
            if track_data['class'] in VEHICLE_CLASSES and len(track_data['positions']) >= 2:
                # Add to vehicles in ROI
                self.vehicles_in_roi.append(track_id)
                
                # If velocity is already calculated, use it for traffic flow analysis
                if track_data['velocity'] is not None:
                    velocity_magnitude = track_data['velocity'][0]
                    self.avg_speeds.append(velocity_magnitude)
        
        # Determine congestion level
        if len(self.vehicles_in_roi) > CONGESTION_THRESHOLD:
            # Check average speed
            if self.avg_speeds:
                avg_speed = sum(self.avg_speeds) / len(self.avg_speeds)
                
                if avg_speed < 15.0:  # Very slow movement (in pixels per second)
                    self.congestion_level = 3  # Heavy
                elif avg_speed < 30.0:  # Slow movement
                    self.congestion_level = 2  # Moderate
                else:
                    self.congestion_level = 1  # Light
            else:
                self.congestion_level = 1  # Light by default
        else:
            self.congestion_level = 0  # No congestion
    
    def detect_lanes(self, frame):
        """Detect lanes in the frame using OpenCV techniques"""
        if not self.enable_lane_detection:
            return frame
            
        try:
            # Convert to grayscale and apply Gaussian blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Define region of interest (lower half of the frame)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            polygon = np.array([
                [(0, height), (0, height*0.6), (width, height*0.6), (width, height)]
            ], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Apply Hough Transform to detect lines
            lines = cv2.HoughLinesP(
                masked_edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=20, 
                minLineLength=40, 
                maxLineGap=50
            )
            
            # Prepare output image for lane visualization
            lane_img = np.zeros_like(frame)
            
            # Group detected lines into left and right lanes
            left_lines = []
            right_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope
                    if x2 - x1 == 0:  # Vertical line
                        continue
                        
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter based on slope
                    if abs(slope) < 0.5:  # Discard nearly horizontal lines
                        continue
                        
                    if slope < 0:  # Negative slope: left lane
                        left_lines.append(line[0])
                    else:  # Positive slope: right lane
                        right_lines.append(line[0])
                
                # Average and extrapolate the lanes
                left_lane = self._average_lane(left_lines, height)
                right_lane = self._average_lane(right_lines, height)
                
                # Draw lanes
                if left_lane is not None:
                    x1, y1, x2, y2 = left_lane
                    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
                
                if right_lane is not None:
                    x1, y1, x2, y2 = right_lane
                    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
                
                # Store current lanes
                self.last_lanes = (left_lane, right_lane)
            elif self.last_lanes:
                # Use previous lanes if no lanes are detected
                left_lane, right_lane = self.last_lanes
                
                if left_lane is not None:
                    x1, y1, x2, y2 = left_lane
                    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
                
                if right_lane is not None:
                    x1, y1, x2, y2 = right_lane
                    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
            
            # Combine lanes with original frame (with transparency)
            return cv2.addWeighted(frame, 1.0, lane_img, 0.5, 0)
            
        except Exception as e:
            print(f"Error in lane detection: {str(e)}")
            return frame
    
    def _average_lane(self, lines, height):
        """Calculate an average lane from multiple detected lines"""
        if not lines:
            return None
            
        # Extract points and calculate average slope and intercept
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Linear regression to find the best fit line
        if len(x_coords) > 0 and len(y_coords) > 0:
            try:
                # Calculate slope and intercept
                polyfit = np.polyfit(y_coords, x_coords, 1)
                slope = polyfit[0]
                intercept = polyfit[1]
                
                # Calculate two points on the line to define the lane
                y1 = height  # Bottom of the frame
                x1 = int(slope * y1 + intercept)
                
                y2 = int(height * 0.6)  # Lower horizon
                x2 = int(slope * y2 + intercept)
                
                return [x1, y1, x2, y2]
            except:
                return None
        return None
    
    def detect_traffic_signs(self, detections):
        """Process traffic sign detections"""
        if not self.enable_traffic_sign:
            return
            
        self.detected_signs = []
        
        for det in detections:
            if det['class'] in TRAFFIC_SIGN_CLASSES and det['confidence'] >= MIN_DETECTION_CONFIDENCE:
                self.detected_signs.append(det)
    
    def process_frame(self, frame):
        """
        Process a single frame with the traffic analysis system.
        
        Args:
            frame: Video frame to process
            
        Returns:
            processed_frame: Frame with visualizations
        """
        try:
            # Store frame dimensions
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Initialize if needed
            if not self.initialized:
                self.initialized = True
            
            # Calculate FPS for velocity calculations
            current_time = time.time()
            time_diff = current_time - self.last_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            self.last_time = current_time
            
            # Increment frame counter
            self.frame_count += 1
            
            # Track processing time
            start_time = time.time()
            
            # Run YOLO detection
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
            
            # Extract detections
            detections = []
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    
                    # Create detection object
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class': class_name,
                        'matched': False
                    }
                    
                    detections.append(detection)
            
            # Update tracking
            self.update_tracks(detections, frame)
            
            # Run specific feature detections
            if self.enable_traffic_sign:
                self.detect_traffic_signs(detections)
            
            # Apply lane detection
            if self.enable_lane_detection:
                frame = self.detect_lanes(frame)
            
            # Visualize results
            processed_frame = self.visualize_results(frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Calculate FPS
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            
            # Add FPS to frame
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return processed_frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
    
    def visualize_results(self, frame):
        """Visualize all detection and analysis results on the frame"""
        try:
            height, width = frame.shape[:2]
            
            # Draw bounding boxes for tracked objects (without labels)
            for track_id, track_data in self.tracks.items():
                if track_data['age'] < MAX_TRACKING_AGE:
                    x1, y1, x2, y2 = track_data['bbox']
                    class_name = track_data['class']
                    
                    # Ensure box is within frame boundaries
                    if not self.is_box_in_frame([x1, y1, x2, y2]):
                        continue
                        
                    # Select color based on object class
                    if class_name in VEHICLE_CLASSES:
                        color = (0, 165, 255)  # Orange
                    elif class_name in PERSON_CLASSES:
                        color = (0, 255, 0)  # Green
                    elif class_name in TRAFFIC_SIGN_CLASSES:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw bounding box without object label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw traffic light state if applicable
                    if class_name == 'traffic light' and track_id in self.traffic_light_states:
                        light_state = self.traffic_light_states[track_id]
                        
                        # Choose color based on traffic light state
                        if light_state == "RED":
                            text_color = (0, 0, 255)  # Red
                        elif light_state == "YELLOW":
                            text_color = (0, 255, 255)  # Yellow
                        elif light_state == "GREEN":
                            text_color = (0, 255, 0)  # Green
                        else:
                            text_color = (255, 255, 255)  # White
                        
                        # Display traffic light state
                        cv2.putText(frame, f"TL: {light_state}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    
                    # Draw velocity vector if available
                    if track_data['velocity'] is not None and class_name in VEHICLE_CLASSES:
                        # Get vector components
                        vel_mag, vel_angle = track_data['velocity']
                        
                        # Calculate vector endpoint
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Scale vector for visualization
                        scale = 0.2
                        end_x = int(center_x + vel_mag * scale * math.cos(vel_angle))
                        end_y = int(center_y + vel_mag * scale * math.sin(vel_angle))
                        
                        # Draw velocity vector
                        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                                       (0, 255, 255), 2, tipLength=0.3)
                        
                        # Display velocity change (acceleration) if available
                        if track_data['acceleration'] is not None:
                            accel = track_data['acceleration']
                            # Only show significant acceleration changes
                            if abs(accel) > 5.0:  # pixels/s²
                                accel_text = f"{accel:.1f} px/s²"
                                accel_color = (0, 255, 0) if accel > 0 else (0, 0, 255)  # Green for acceleration, Red for deceleration
                                cv2.putText(frame, accel_text, (center_x, center_y + 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, accel_color, 2)
            
            # Add traffic flow and congestion information
            if self.enable_traffic_flow:
                # Determine congestion color
                if self.congestion_level == 0:
                    cong_color = (0, 255, 0)  # Green
                    cong_text = "No Congestion"
                elif self.congestion_level == 1:
                    cong_color = (0, 255, 255)  # Yellow
                    cong_text = "Light Congestion"
                elif self.congestion_level == 2:
                    cong_color = (0, 165, 255)  # Orange
                    cong_text = "Moderate Congestion"
                else:
                    cong_color = (0, 0, 255)  # Red
                    cong_text = "Heavy Congestion"
                
                # Display congestion level
                cv2.putText(frame, cong_text, (10, height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cong_color, 2)
                
                # Display vehicle count in ROI
                cv2.putText(frame, f"Vehicles in view: {len(self.vehicles_in_roi)}", 
                            (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Display average speed if available (in pixels per second)
                if self.avg_speeds:
                    avg_speed = sum(self.avg_speeds) / len(self.avg_speeds)
                    cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} px/s", 
                                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add vehicle counting information
            if self.enable_vehicle_counting:
                # Display total count
                cv2.putText(frame, f"Total Vehicles: {self.total_vehicles}", 
                            (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display counts by type
                y_offset = 60
                for vehicle_class, count in self.vehicle_counts.items():
                    if count > 0:
                        cv2.putText(frame, f"{vehicle_class}: {count}", 
                                    (width - 250, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30
            
            return frame
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return frame
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video file with all enabled analysis features.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (None to not save)
        """
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
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
            
            # Store frame dimensions
            self.frame_width = width
            self.frame_height = height
            
            print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} total frames")
            
            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                print(f"Output will be saved to: {output_path}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print("Processing video... Press 'q' to quit.")
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Process frame with traffic analysis system
                processed_frame = self.process_frame(frame)
                
                # Calculate progress
                progress = (self.frame_count / total_frames) * 100
                cv2.putText(processed_frame, f"Progress: {progress:.1f}%", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the processed frame
                cv2.imshow("Traffic Analysis", processed_frame)
                
                # Write frame to output video if specified
                if writer:
                    writer.write(processed_frame)
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count}/{total_frames} frames ({progress:.1f}%)")
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
            
            # Release resources
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print performance statistics
            self._print_statistics()
            
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
            # Clean up resources
            try:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                if 'writer' in locals() and writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def _print_statistics(self):
        """Print performance and traffic statistics"""
        print("\n" + "="*50)
        print("TRAFFIC ANALYSIS RESULTS")
        print("="*50)
        
        # Performance stats
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\nPerformance Statistics:")
            print(f"  Total frames processed: {self.frame_count}")
            print(f"  Average processing time: {avg_time:.4f} seconds per frame")
            print(f"  Average FPS: {avg_fps:.2f}")
        
        # Vehicle counting stats
        if self.enable_vehicle_counting:
            print(f"\nVehicle Counting Statistics:")
            print(f"  Total vehicles: {self.total_vehicles}")
            
            for vehicle_class, count in self.vehicle_counts.items():
                if count > 0:
                    percentage = (count / self.total_vehicles) * 100 if self.total_vehicles > 0 else 0
                    print(f"  {vehicle_class}: {count} ({percentage:.1f}%)")
        
        # Traffic flow stats
        if self.enable_traffic_flow and self.avg_speeds:
            avg_speed = sum(self.avg_speeds) / len(self.avg_speeds)
            
            print(f"\nTraffic Flow Statistics:")
            print(f"  Average vehicle speed: {avg_speed:.2f} pixels/second")
            
            congestion_text = "None"
            if self.congestion_level == 1:
                congestion_text = "Light"
            elif self.congestion_level == 2:
                congestion_text = "Moderate"
            elif self.congestion_level == 3:
                congestion_text = "Heavy"
            
            print(f"  Final congestion level: {congestion_text}")
            print(f"  Vehicles in ROI at end: {len(self.vehicles_in_roi)}")
        
        # Traffic sign stats
        if self.enable_traffic_sign:
            print(f"\nTraffic Sign Statistics:")
            sign_counts = {}
            for sign in self.detected_signs:
                sign_class = sign['class']
                sign_counts[sign_class] = sign_counts.get(sign_class, 0) + 1
            
            for sign_class, count in sign_counts.items():
                print(f"  {sign_class}: {count}")
                
            # Traffic light states
            tl_states = {"RED": 0, "YELLOW": 0, "GREEN": 0, "UNKNOWN": 0}
            for state in self.traffic_light_states.values():
                if state in tl_states:
                    tl_states[state] += 1
            
            print(f"  Traffic light states:")
            for state, count in tl_states.items():
                if count > 0:
                    print(f"    {state}: {count}")
        
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Create the traffic analysis system
    system = TrafficAnalysisSystem(
        model_path="yolov8n.pt",
        enable_vehicle_counting=ENABLE_VEHICLE_COUNTING,
        enable_traffic_flow=ENABLE_TRAFFIC_FLOW,
        enable_traffic_sign=ENABLE_TRAFFIC_SIGN,
        enable_lane_detection=ENABLE_LANE_DETECTION
    )
    
    # Process a video file
    video_path = "sample/traffic_bus.mp4"  # Using the specified video path
    output_path = "analyzed_traffic.mp4"   # Replace with desired output path
    
    system.process_video(video_path, output_path)