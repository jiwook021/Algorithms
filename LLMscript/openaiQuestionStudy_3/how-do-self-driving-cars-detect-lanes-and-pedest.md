Self-driving cars use a combination of sensors and clever software to see lanes and pedestrians.  Let's break it down:

**1. Sensing the World:**

* **Cameras:** Think of these like your eyes.  Multiple cameras around the car take pictures of everything around it. These pictures are incredibly detailed and taken very frequently.

* **Lidar (Light Detection and Ranging):** This is like a super-powered laser rangefinder. It sends out laser beams and measures how long it takes for them to bounce back. This tells the car the distance to objects, creating a 3D map of its surroundings.  Think of it as seeing in "depth".

* **Radar (Radio Detection and Ranging):** Similar to lidar, but uses radio waves instead of lasers. It's good at detecting objects in bad weather (rain, fog) where lidar might struggle. It's also better at detecting the speed of objects.

* **Ultrasonic Sensors:** These are like tiny sonar devices, used primarily for very close-range detection, such as parking assistance or detecting objects very close to the car's bumpers.


**2. Processing the Information:**

* **Image Recognition:** The car's computer uses sophisticated software to analyze the images from the cameras.  This software is trained on millions of images to recognize things like lane markings (white lines, dashed lines, solid lines), pedestrians (people walking, running, standing), cars, traffic lights, and signs.  It's like teaching a computer to "see" and understand what it sees.

* **Point Cloud Processing:** The lidar and radar data is also processed.  This data is a 3D point cloud showing the location and distance of every object detected. The computer combines this with the camera data for a more complete picture.

* **Data Fusion:** The information from all the sensors (cameras, lidar, radar, ultrasonic) is combined.  This helps the car get a more accurate and robust understanding of its environment, dealing with the limitations of each individual sensor.  For instance, lidar might miss a pedestrian behind a tree, but the camera might see them.


**3. Understanding the Scene:**

* **Lane Detection:** The computer uses the processed images and point cloud data to identify the lane markings on the road. It determines the car's position within the lane and the curvature of the road.

* **Pedestrian Detection:**  The computer uses image recognition to find pedestrians.  It looks for shapes, movements, and other characteristics consistent with a human being.  The data from lidar and radar helps to determine the distance and speed of the pedestrian.

**In short:** Self-driving cars "see" their surroundings using multiple sensors, and then use powerful computers and clever software to interpret that information, figuring out where lanes are and identifying pedestrians to navigate safely.  It's a complex process, but it boils down to sensing, processing, and understanding.
