#!/usr/bin/env python

import cv2
import numpy as np
from cv2 import aruco
import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_matrix
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray

'''
Example of Aruco in Stonefish:
<static name="aruco1" type="box">
    <dimensions xyz="0.1875 0.1875 0.001" />
    <origin rpy="${pi/2} 0.0 0.0" xyz="0.0 0.0 0.0" />
    <material name="Plastic" />
    <look name="aruco1" uv_mode="3" />
    <world_transform rpy="0.0 0.0 0.0" xyz="1.4 -0.2 -0.15" />
</static>

dictionary: DICT_ARUCO_ORIGINAL
aruco img: turtlebot_simulation/resources/textures
'''


class Aruco_detector:
    def __init__(self):
        # Type of testing
        self.physical = True
        # Image configurations and detector params
        self.bridge = CvBridge()
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters_create()
        if self.physical:
            self.marker_size = 0.193
        else:
            self.marker_size = 0.18
        cam_info_msg = rospy.wait_for_message("/turtlebot/kobuki/realsense/color/camera_info", CameraInfo)
        self.camera_matrix = np.array(cam_info_msg.K).reshape(3,3)
        self.dist_coeffs = np.array(cam_info_msg.D)
        # self.camera_matrix = np.array([[800, 0, 320],
        #                                 [0, 800, 240],
        #                                 [0, 0, 1]])
        # self.dist_coeffs = np.array([0, 0, 0, 0, 0])

        if self.physical:
            # Physical robot sub:
            self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_raw/compressed", CompressedImage, self.img_callback) # input
        else:
            # Simulation sub:
            self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.img_callback) # input
        
        self.img_aruco_pub = rospy.Publisher("/turtlebot/kobuki/SLAM/aruco/image", Image, queue_size=10)     # feedback processing
        self.markers_pub = rospy.Publisher("/turtlebot/kobuki/SLAM/aruco/markers", MarkerArray, queue_size=10)  # aruco poses w.r.t. robot

        rospy.loginfo("Aruco detector node started")
        

    def img_callback(self, msg):
        if self.physical:
            try:
                np_data = np.frombuffer(msg.data, np.uint8)
                img_np = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                # debug conversion
                # img_msg = self.bridge.cv2_to_imgmsg(img_np, encoding='bgr8')
                # img_msg.header.stamp = msg.header.stamp
                # img_msg.header.frame_id = "camera_color_optical_frame"
                # self.img_aruco_pub.publish(img_msg)
                cv_image = img_np
                # rospy.loginfo("Received image")
            except CvBridgeError as e:
                rospy.logerr(e)
                return
        else:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # rospy.loginfo("Received image")
            except CvBridgeError as e:
                rospy.logerr(e)
                return

        # Detect corners
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # Upon detection
        if ids is not None:
            # Draw markers detector
            aruco.drawDetectedMarkers(cv_image, corners, ids)

            # Get transforms
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            # Several markers publishing
            markers_array = MarkerArray()

            for i in range(len(ids)):
                marker = Marker()
                # marker.header.stamp = rospy.Time.now()
                if self.physical:
                    marker.header.frame_id = "realsense_color_optical_frame"
                else:
                    marker.header.frame_id = "camera_color_optical_frame"
                marker.id = ids[i][0]
                marker.type = Marker.SPHERE
                marker.color.a = 1.0
                marker.color.b = 0.0
                marker.color.g = 0.0
                marker.color.r = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]
                # Convert rotation vector to quaternion
                R, _ = cv2.Rodrigues(rvecs[i][0])
                q = quaternion_from_matrix(np.vstack([
                        np.hstack([R, np.zeros((3, 1))]),
                        [0, 0, 0, 1]
                    ]))
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]
                marker.pose = pose

                markers_array.markers.append(marker)
            self.markers_pub.publish(markers_array)
            # print('markers published')
        
        # Publish detector
        out = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.img_aruco_pub.publish(out)

if __name__ == '__main__':
    rospy.init_node('aruco_detector')
    detector = Aruco_detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Aruco detector: shutting down")
    cv2.destroyAllWindows()