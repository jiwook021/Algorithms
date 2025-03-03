#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg
import tf2_ros

class LaserToPointCloud:
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)

    def scan_callback(self, scan_msg):
        try:
            # Convert LaserScan to PointCloud2
            pc2_msg = self.lp.projectLaser(scan_msg)
            # Publish the point cloud
            self.pc_pub.publish(pc2_msg)
        except Exception as e:
            rospy.logerr("Error converting laser scan to point cloud: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('laser_to_pointcloud', anonymous=True)
    node = LaserToPointCloud()
    rospy.spin()

#chmod +x laser_to_pointcloud.py
#rosrun PID_drone_control laser_to_pointcloud.py
#PointCloud2 /point_cloud