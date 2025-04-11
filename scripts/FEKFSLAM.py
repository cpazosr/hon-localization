#!/usr/bin/python

import numpy as np
import rospy
import threading
import queue
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class FEKFSLAM:
    
    def __init__(self):
        rospy.init_node("FEKFSLAM")

        # Odom frames
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        # Message queues
        self.odom_queue = queue.Queue()
        self.imu_queue = queue.Queue()
        
        # Sensor states
        # IMU
        self.latest_imu = None
        self.latest_imu_time = None
        self.imu_timeout = 2*(1.0/10.0)   # freq = 10 Hz
        v_yaw_std = np.deg2rad(5)   # 5 degrees
        self.yaw_Rm = np.array([v_yaw_std**2])
        self.Hm = np.array([[0,0,1]])
        self.Vm = np.eye(self.Hm.shape[0])
        # Features
        self.features_timeout = 2*(1.0/1)   # freq = ...
        self.yaw = None

        # Subscribers
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data", Imu, self.imu_callback)    # Imu - 10 Hz
        self.odom_sub = rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_callback)     # robot odom - 100 Hz

        # Publishers
        self.odom_pub = rospy.Publisher("/SLAM/odom", Odometry, queue_size=20)

        # Node thread execution
        self.thread = threading.Thread(target=self.main)
        self.thread.daemon = True
        self.thread.start()

    def imu_callback(self, msg):
        # Obtain the orientation of robot XY plane delivered from the IMU
        # Applying queue to not miss a message but still check the message is recent 
        # Applying old message updates is dangerous as the filter needs closest/realest values for update corrections
        q = msg.orientation
        roll, pitch ,self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        """cov  = msg.orientation_covariance
        # self.yaw_Rm = cov[-1]    # from 3x3 mat (roll,pitch,yaw) -> element at [2,2]
        self.imu_queue.put((yaw,rospy.Time.now()))    # compass orientation [euler], time 
        print ( "imu callback:", yaw)"""

    def odom_callback(self, msg):
        # Obtain the odometry published by the differential drive robot
        # Because we are integrating we need to use every message, so a queue is ideal
        try:
            # Prediction
            odom_msg = msg
            x = odom_msg.pose.pose.position.x
            y = odom_msg.pose.pose.position.y
            q = odom_msg.pose.pose.orientation
            _,_,theta = euler_from_quaternion([q.x,q.y,q.z,q.w])
            xk_bar = np.array([[x],[y],[theta]])
            cov = np.array(odom_msg.pose.covariance).reshape(6, 6)
            Pk_bar = np.array([
                [cov[0, 0], cov[0, 1], cov[0, 5]],  # x
                [cov[1, 0], cov[1, 1], cov[1, 5]],  # y
                [cov[5, 0], cov[5, 1], cov[5, 5]]   # yaw
            ])
            # print("Prediction")
            
            # Upon new IMU data only
            try:
                #zk, imu_msg_time = self.imu_queue.get_nowait()
                if (rospy.Time.now() - imu_msg_time).to_sec() < self.imu_timeout:
                    # Update with IMU
                    # print("IMU received")
                    xk, Pk = self.Update(zk, self.yaw_Rm, xk_bar, Pk_bar, self.Hm, self.Vm)
            except queue.Empty:     # no IMU message
                xk = xk_bar
                Pk = Pk_bar
        
            # Publish
            self.publishOdometry(xk, Pk)
            
        except queue.Empty: # no odom message
            print("empty odom")
            pass
            

    def h(self, xk_bar):
        # Actual state yaw of the robot
        return xk_bar[2]

    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk):
        # Perform Kalman equation for Update
        # print("zk:",zk,"Rk:",Rk,"xk_bar:",xk_bar,"Pk_bar:",Pk_bar,"Hk:",Hk,"Vk:",Vk)
        Kk = Pk_bar@Hk.T @ np.linalg.pinv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T) # Kalman Gain [3x1] -> pseudoinverse works for 1x1 and bigger matrices
        xk = xk_bar + Kk@np.atleast_2d(zk - self.h(xk_bar)).reshape(-1,1) # updated state [3x1] -> atleast2d ensures compatibility with 1x1 and bigger matrices
        I = np.eye(Pk_bar.shape[0])
        Pk = (I - Kk@Hk) @ Pk_bar @ (I - Kk@Hk).T
        
        return xk, Pk

    def publishOdometry(self, xk, Pk):
        # Publish Odometry message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = xk[0]
        odom.pose.pose.position.y = xk[1]
        odom.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, xk[2])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
                                        # x            y        z  roll pitch   yaw
        odom.pose.covariance = np.array([Pk[0, 0],  Pk[0, 1],   0,  0,  0,      Pk[0,2],  # x
                                         Pk[1, 0],  Pk[1, 1],   0,  0,  0,      Pk[1,2],  # y
                                         0,         0,          0,  0,  0,      0,        # z
                                         0,         0,          0,  0,  0,      0,        # roll
                                         0,         0,          0,  0,  0,      0,        # pitch
                                         Pk[2, 0],  Pk[2, 1],   0,  0,  0,      Pk[2,2]]) # yaw

        # odom.twist.twist.linear.x = v
        # odom.twist.twist.angular.z = w

        self.odom_pub.publish(odom)

    def main(self):
        # Main node execution
        rate = rospy.Rate(50)   # 50 Hz
        while not rospy.is_shutdown():
            
            rate.sleep()

if __name__ == '__main__':
    try:
        SLAMNode = FEKFSLAM()
        rospy.spin()    # only keeps subscribers alive but execution is respected by the thread in the node
    except Exception as e:
        print(e)