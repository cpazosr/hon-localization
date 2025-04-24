#!/usr/bin/python

import numpy as np
import rospy
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState, Imu
from tf.broadcaster import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
class DifferentialDrive:
    def __init__(self) -> None:
        
        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.wheel_vel_noise = np.array([0.01, 0.01])

        # get params
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        self.left_wheel_name = rospy.get_param("~wheel_left_joint_name", "kobuki/wheel_left_joint")
        self.right_wheel_name = rospy.get_param("~wheel_right_joint_name", "kobuki/wheel_right_joint")

        rospy.loginfo("Left wheel joint name: %s", self.left_wheel_name)
        rospy.loginfo("Right wheel joint name: %s", self.right_wheel_name)
        rospy.loginfo("Odom frame: %s", self.odom_frame)
        rospy.loginfo("Base frame: %s", self.base_frame)

        # pose of the robot (originally coming from dead reckoning)
        self.x = 3.0 #3.0 -0.78 -0.2
        self.y = -0.78
        self.th = np.pi / 2.0

        
        self.P = np.diag([0.1, 0.001, 0.01])
        
        #ekf state (made by me)
        self.eta_k = np.array([0,0,0]).reshape(-1, 1)      
        self.Pk = np.diag([0.1, 0.001, 0.01])
        
        
        # velocity and angular velocity of the robot
        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        self.left_wheel_received = False

        self.first_received = False
        self.last_time = rospy.Time.now()
        
        v_yaw_std = np.deg2rad(5)   # 5 degrees

        self.yaw_Rm = np.array([v_yaw_std**2])      #covariance mat from std deviation
        self.Hm = np.array([[0,0,1]])
        self.Vm = np.eye(self.Hm.shape[0])

        # joint state subscriber
        self.js_sub = rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        self.state_pub = rospy.Publisher("state_ekf", Odometry, queue_size=20)
        
        
        # imu subscriber
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data", Imu, self.imu_callback)

        # odom publisher
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=20)
        self.tf_br = TransformBroadcaster()
        rospy.Timer(rospy.Duration(0.1), self.publish_state)
        
        

    def f(self, dt, wheel_velocities):

        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0
        w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
        
        x_k = self.x + np.cos(self.th) * v * dt
        y_k = self.y + np.sin(self.th) * v * dt
        th_k = self.th + w * dt

        return np.array([x_k, y_k, th_k])
    
    def Jfx(self, dt, wheel_velocities):

        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0

        J = np.array([[1, 0, -v * np.sin(self.th) * dt],
                      [0, 1, v * np.cos(self.th) * dt],
                      [0, 0, 1]])
        
        return J
    
    def Jfw(self, dt):
            
        J = np.array([[np.cos(self.th) * dt / 2.0, np.cos(self.th) * dt / 2.0],
                        [np.sin(self.th) * dt / 2.0, np.sin(self.th) * dt / 2.0],
                        [dt / self.wheel_base_distance, -dt / self.wheel_base_distance]])
        
        return J
    
    def prediction(self, dt, wheel_velocities):

        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0
        w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
        
        x_k = self.eta_k[0,0] + np.cos(self.th) * v * dt
        y_k = self.eta_k[1,0] + np.sin(self.th) * v * dt
        th_k = self.eta_k[2,0] + w * dt

        # propagate covariance
        Pk = self.Jfx_k(dt, wheel_velocities) @ self.Pk @ self.Jfx_k(dt, wheel_velocities).T  + self.Jfw_k(dt) @ np.diag(self.wheel_vel_noise) @ self.Jfw_k(dt).T   
        return np.array([x_k, y_k, th_k]).reshape(-1, 1), Pk
    
    def Jfx_k(self, dt, wheel_velocities):

        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0

        J = np.array([[1, 0, -v * np.sin(self.eta_k[2,0]) * dt],
                      [0, 1, v * np.cos(self.eta_k[2,0]) * dt],
                      [0, 0, 1]])
        
        return J
    
    
    def Jfw_k(self, dt):
            
        J = np.array([[np.cos(self.eta_k[2,0]) * dt / 2.0, np.cos(self.eta_k[2,0]) * dt / 2.0],
                        [np.sin(self.eta_k[2,0]) * dt / 2.0, np.sin(self.eta_k[2,0]) * dt / 2.0],
                        [dt / self.wheel_base_distance, -dt / self.wheel_base_distance]])
        
        return J

            
    def publish_state(self, event):
        print("Publish state")
        state  = Odometry()
        state.header.stamp = rospy.Time.now()  
        state.header.frame_id = self.odom_frame
        state.child_frame_id = self.base_frame
        state.pose.pose.position.x = self.eta_k[0,0]
        state.pose.pose.position.y = self.eta_k[1,0]
        state.pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, self.eta_k[2,0])
        state.pose.pose.orientation.x = q[0]
        state.pose.pose.orientation.y = q[1]

        state.pose.pose.orientation.z = q[2]
        state.pose.pose.orientation.w = q[3]
        state.pose.covariance = np.array([self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0,2],
                                                 self.Pk[1, 0], self.Pk[1, 1], 0, 0, 0, self.Pk[1,2],
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2,2]])
                                        
        self.state_pub.publish(state)
    

    
    def joint_state_callback(self, msg):

        if msg.name[0] == self.left_wheel_name:
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_received = True
            return
        elif msg.name[0] == self.right_wheel_name:
            self.right_wheel_velocity = msg.velocity[0]
            if (self.left_wheel_received):
                if (not self.first_received):
                    self.first_received = True
                    self.last_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
                    return
                
                #calculate dt
                current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
                dt = (current_time - self.last_time).to_sec()
                self.last_time = current_time

                # integrate position
                wheel_velocities = np.array([self.left_wheel_velocity, self.right_wheel_velocity])
                [self.x, self.y, self.th] = self.f(dt, wheel_velocities)

                # propagate covariance
                self.P = self.Jfx(dt, wheel_velocities) @ self.P @ self.Jfx(dt, wheel_velocities).T  + self.Jfw(dt) @ np.diag(self.wheel_vel_noise) @ self.Jfw(dt).T

                # Calculate velocity
                lin_vel = wheel_velocities * self.wheel_radius
                v = (lin_vel[0] + lin_vel[1]) / 2.0
                w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
                
                # Predict state
                self.eta_k, self.Pk = self.prediction(dt, wheel_velocities)

                # Reset flag
                self.left_wheel_received = False

                # Publish odom

                odom = Odometry()
                odom.header.stamp = current_time
                odom.header.frame_id = self.odom_frame
                odom.child_frame_id = self.base_frame

                odom.pose.pose.position.x = self.x
                odom.pose.pose.position.y = self.y
                odom.pose.pose.position.z = 0.0

                q = quaternion_from_euler(0, 0, self.th)
                odom.pose.pose.orientation.x = q[0]
                odom.pose.pose.orientation.y = q[1]
                odom.pose.pose.orientation.z = q[2]
                odom.pose.pose.orientation.w = q[3]

                odom.pose.covariance = np.array([self.P[0, 0], self.P[0, 1], 0, 0, 0, self.P[0,2],
                                                 self.P[1, 0], self.P[1, 1], 0, 0, 0, self.P[1,2],
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 self.P[2, 0], self.P[2, 1], 0, 0, 0, self.P[2,2]])
                                                 

                odom.twist.twist.linear.x = v
                odom.twist.twist.angular.z = w

                self.odom_pub.publish(odom)

                self.tf_br.sendTransform((self.x, self.y, 0.0), q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)

# publish with timer - it is already in self.eta k and self.Pk
# to do update: add IMU callback AND updateeeeee function and value of self.eta_k and self.Pk 
# dont use selfyaw just yaw 

    def imu_callback(self, msg):
        # Obtain the orientation of robot XY plane delivered from the IMU
        # Applying queue to not miss a message but still check the message is recent 
        # Applying old message updates is dangerous as the filter needs closest/realest values for update corrections
        q = msg.orientation
        roll, pitch ,self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = wrap_angle(self.yaw -np.pi/2) # to align with odom frame
        # self.yaw_Rm = cov[-1]    # from 3x3 mat (roll,pitch,yaw) -> element at [2,2]
        print( "imu callback:", self.yaw)
        print(self.eta_k)
        self.eta_k, self.Pk = self.Update_imu(self.yaw, self.yaw_Rm, self.eta_k.copy(), self.Pk.copy(), self.Hm, self.Vm)
    
    def Update_imu(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk):
        # Perform Kalman equation for Update
        # print("zk:",zk,"Rk:",Rk,"xk_bar:",xk_bar,"Pk_bar:",Pk_bar,"Hk:",Hk,"Vk:",Vk)
        Kk = Pk_bar@Hk.T @ np.linalg.pinv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T) # Kalman Gain [3x1] -> pseudoinverse works for 1x1 and bigger matrices
        # xk = xk_bar + Kk@np.atleast_2d(zk - self.h(xk_bar)).reshape(-1,1) # updated state [3x1] -> atleast2d ensures compatibility with 1x1 and bigger matrices
        
        print(Kk.reshape(3,1) @ wrap_angle(zk - self.h(xk_bar)).reshape(-1,1) )

        eta_k = xk_bar + Kk.reshape(3,1) @ wrap_angle(zk - self.h(xk_bar)).reshape(-1,1) # updated state [3x1] -> atleast2d ensures compatibility with 1x1 and bigger matrices
        

        I = np.eye(Pk_bar.shape[0])
        Pk = (I - Kk@Hk) @ Pk_bar @ (I - Kk@Hk).T
        
        return eta_k.reshape(-1,1), Pk
    
    def h(self, xk_bar):
        # Actual state yaw of the robot
        return xk_bar[2]


if __name__ == '__main__':

    rospy.init_node("differential_drive")

    robot = DifferentialDrive()

    rospy.spin()

    





