import cv2
import cv2.aruco as aruco

# Define the dictionary (e.g., 6x6 with 250 markers)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_id = 0  # Choose an ID (0-249 for this dictionary)
marker_size = 700  # Pixels for the marker
border_size = 100  # Pixels for white border

# Generate the marker
marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Add white border
marker_img_border = cv2.copyMakeBorder(marker_img, border_size, border_size, border_size, border_size, 
                                       cv2.BORDER_CONSTANT, value=255)

# Save the image
cv2.imwrite("aruco_marker_0.png", marker_img_border)
print("ArUco marker saved as aruco_marker_0.png")
# python3 generate_aruco.py

# mkdir -p ~/.gazebo/models/aruco_marker_0/materials/textures
# mkdir -p ~/.gazebo/models/aruco_marker_0/materials/scripts
# cp aruco_marker_0.png ~/.gazebo/models/aruco_marker_0/materials/textures/
# vim ~/.gazebo/models/aruco_marker_0/materials/scripts/aruco.material
# material aruco_marker_0
# {
#   technique
#   {
#     pass
#     {
#       texture_unit
#       {
#         texture ../textures/aruco_marker_0.png
#       }
#     }
#   }
# }
# vim ~/.gazebo/models/aruco_marker_0/model.sdf


# vim ~/.gazebo/models/aruco_marker_0/model.sdf
# <?xml version="1.0"?>
# <sdf version="1.6">
#   <model name="aruco_marker_0">
#     <static>true</static>
#     <link name="link">
#       <visual name="visual">
#         <geometry>
#           <box>
#             <size>0.2 0.2 0.001</size> <!-- Size in meters: 20cm x 20cm x 0.1cm -->
#           </box>
#         </geometry>
#         <material>
#           <script>
#             <uri>file://materials/scripts/aruco.material</uri>
#             <name>aruco_marker_0</name>
#           </script>
#         </material>
#       </visual>
#       <collision name="collision">
#         <geometry>
#           <box>
#             <size>0.2 0.2 0.001</size>
#           </box>
#         </geometry>
#       </collision>
#     </link>
#   </model>
# </sdf>

# export GAZEBO_MODEL_PATH=~/.gazebo/models:$GAZEBO_MODEL_PATH

# rosrun gazebo_ros spawn_model -file ~/.gazebo/models/aruco_marker_0/model.sdf -sdf -model aruco_marker_0 -x 1.0 -y 1.0 -z 0.0






# python3 generate_aruco.py
# mkdir -p ~/.gazebo/models/aruco_marker_0/materials/textures;
# mkdir -p ~/.gazebo/models/aruco_marker_0/materials/scripts;
# cp aruco_marker_0.png ~/.gazebo/models/aruco_marker_0/materials/textures/;
# echo -e "material aruco_marker_0\n{\n  technique\n  {\n    pass\n    {\n      texture_unit\n      {\n        texture ../textures/aruco_marker_0.png\n      }\n    }\n  }\n}" > ~/.gazebo/models/aruco_marker_0/materials/scripts/aruco.material;
# echo -e '<?xml version="1.0"?>\n<sdf version="1.6">\n  <model name="aruco_marker_0">\n    <static>true</static>\n    <link name="link">\n      <visual name="visual">\n        <geometry>\n          <box>\n            <size>0.2 0.2 0.001</size>\n          </box>\n        </geometry>\n        <material>\n          <script>\n            <uri>file://materials/scripts/aruco.material</uri>\n            <name>aruco_marker_0</name>\n          </script>\n        </material>\n      </visual>\n      <collision name="collision">\n        <geometry>\n          <box>\n            <size>0.2 0.2 0.001</size>\n          </box>\n        </geometry>\n      </collision>\n    </link>\n  </model>\n</sdf>' > ~/.gazebo/models/aruco_marker_0/model.sdf;
# echo -e '<?xml version="1.0"?>\n<model>\n  <name>aruco_marker_0</name>\n  <version>1.0</version>\n  <sdf>model.sdf</sdf>\n  <author><name>Jiwook</name></author>\n  <description>ArUco Marker with ID 0</description>\n</model>' > ~/.gazebo/models/aruco_marker_0/model.config
# roslaunch hector_quadrotor_demo outdoor_flight_gazebo.launch
# rosrun gazebo_ros spawn_model -file ~/.gazebo/models/aruco_marker_0/model.sdf -sdf -model aruco_marker_0 -x 1.0 -y 1.0 -z 0.0