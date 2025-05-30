#!/usr/bin/python

import numpy as np
import rospy
import scipy
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseWithCovariance
from localization.msg import ArucoWithCovariance, ArucoWithCovarianceArray
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState, Imu
from tf.broadcaster import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import Pose as P
import Feature as F
import MapFeature as MF
# from MapFeature import g

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class DifferentialDrive:
    def __init__(self) -> None:
        
        # Testing type
        self.physical = True

        # Robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.wheel_vel_noise = 5*np.array([0.01, 0.01])

        # Get params
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "turtlebot/kobuki/base_footprint")#"base_footprint")
        
        if self.physical:
            # Physical robot:
            self.left_wheel_name = rospy.get_param("~wheel_left_joint_name", "turtlebot/kobuki/wheel_left_link")
            self.right_wheel_name = rospy.get_param("~wheel_right_joint_name", "turtlebot/kobuki/wheel_right_link")
        # Simulation:
        else:
            self.left_wheel_name = rospy.get_param("~wheel_left_joint_name", "kobuki/wheel_left_joint")
            self.right_wheel_name = rospy.get_param("~wheel_right_joint_name", "kobuki/wheel_right_joint")
        
        # Logging
        rospy.loginfo("Left wheel joint name: %s", self.left_wheel_name)
        rospy.loginfo("Right wheel joint name: %s", self.right_wheel_name)
        rospy.loginfo("Odom frame: %s", self.odom_frame)
        rospy.loginfo("Base frame: %s", self.base_frame)

        # Inital state
        if self.physical:
            # Physical: 
            self.x = 0.0
            self.y = 0.0
            self.th = 0.0
        else:
            # Simulation:
            self.x = 3.0 #3.0 -0.78 -0.2
            self.y = -0.78
            self.th = np.pi/2.0
        
        self.xk = np.array([self.x, self.y, self.th]).reshape(3,1)
        self.P = 0.1*np.diag([0.1, 0.1, 0.01])
        self.feature_ids = []
        
        # EKF state
        # self.eta_k = np.array([0,0,0]).reshape(-1, 1)
        # self.Pk = np.diag([0.1, 0.001, 0.01])
        self.xF_dim = 2
        self.zfi_dim = 2
        self.xB_dim = 2
        self.xBpose_dim = 3
        self.nf = 0
        self.alpha = 0.95
        
        # velocity and angular velocity of the robot
        self.lin_vel = 0.0
        self.ang_vel = 0.0

        # Joint states flags
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False
        self.first_received = False
        self.last_time = rospy.Time.now()
        
        # IMU
        v_yaw_std = np.deg2rad(5)   # 5 degrees
        self.yaw_Rm = np.array([v_yaw_std**2])      # or alternatively the covariance mat from std deviation
        self.Hm = np.array([[0,0,1]])
        self.Vm = np.eye(self.Hm.shape[0])
        self.imu_c = 0
        self.imu_update_freq = 10
        # Aruco Markers
        self.xy_Rk = 0.5*np.diag(np.array([1**2, 1**2])) # covariance of reading a coordinate feature
        self.aruco_c = 0
        self.aruco_update_freq = 10

        # Sensors subscribers
        if self.physical:
            # Physical:
            self.js_phys_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.joint_state_callback_physical)
        else:
           # Simulation
            self.js_sim_sub = rospy.Subscriber("joint_states", JointState, self.joint_state_callback_sim)
     
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data", Imu, self.imu_callback)
        self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/SLAM/aruco/markers", MarkerArray, self.markers_callback)
        # Odom publisher
        # self.state_pub = rospy.Publisher("/turtlebot/kobuki/SLAM/EKF_odom", Odometry, queue_size=20)
        self.odom_pub = rospy.Publisher("/turtlebot/kobuki/SLAM/odom", Odometry, queue_size=20)    # /turtlebot/kobuki/odom
        self.markers_pub = rospy.Publisher("/turtlebot/kobuki/SLAM/features", ArucoWithCovarianceArray, queue_size=1)
        self.tf_br = TransformBroadcaster()
        self.pose_publishers = {
            1: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_1", Odometry, queue_size=1),
            11: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_11", Odometry, queue_size=1),
            21: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_21", Odometry, queue_size=1),
            31: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_31", Odometry, queue_size=1),
            41: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_41", Odometry, queue_size=1),
            51: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_51", Odometry, queue_size=1),
            61: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_61", Odometry, queue_size=1),
            71: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_71", Odometry, queue_size=1),
            81: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_81", Odometry, queue_size=1),
            91: rospy.Publisher("/turtlebot/kobuki/SLAM/markers/pose_cov_91", Odometry, queue_size=1)
        }
        # rospy.Timer(rospy.Duration(0.1), self.publish_state)
        rospy.Timer(rospy.Duration(0.1), self.publish_features)


    def f(self, dt, wheel_velocities):
        ''' Motion model '''
        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0
        w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
        
        x_k = self.x + np.cos(self.th) * v * dt
        y_k = self.y + np.sin(self.th) * v * dt
        th_k = self.th + w * dt

        return np.array([x_k, y_k, th_k])
    
    def Jfx(self, dt, wheel_velocities):
        ''' Motion model Jacobian w.r.t. the state '''
        lin_vel = wheel_velocities * self.wheel_radius

        v = (lin_vel[0] + lin_vel[1]) / 2.0

        J = np.array([[1, 0, -v * np.sin(self.th) * dt],
                      [0, 1, v * np.cos(self.th) * dt],
                      [0, 0, 1]])
        
        return J
    
    def Jfw(self, dt):
        ''' Motion model Jacobian w.r.t. noise '''
        J = np.array([[np.cos(self.th) * dt / 2.0, np.cos(self.th) * dt / 2.0],
                        [np.sin(self.th) * dt / 2.0, np.sin(self.th) * dt / 2.0],
                        [dt / self.wheel_base_distance, -dt / self.wheel_base_distance]])
        
        return J
            
    def publish_state(self, event):
        # TODO: consider only updating state and publishing every 100Hz
        state  = Odometry()
        state.header.stamp = rospy.Time.now()  
        state.header.frame_id = self.odom_frame
        state.child_frame_id = self.base_frame
        state.pose.pose.position.x = self.x # self.eta_k[0,0]
        state.pose.pose.position.y = self.y # self.eta_k[1,0]
        state.pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, self.th) # self.eta_k[2,0])
        state.pose.pose.orientation.x = q[0]
        state.pose.pose.orientation.y = q[1]

        state.pose.pose.orientation.z = q[2]
        state.pose.pose.orientation.w = q[3]
        state.pose.covariance = np.array([self.P[0, 0], self.P[0, 1], 0, 0, 0, self.P[0,2],
                                                 self.P[1, 0], self.P[1, 1], 0, 0, 0, self.P[1,2],
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0,
                                                 self.P[2, 0], self.P[2, 1], 0, 0, 0, self.P[2,2]])
        # state.pose.covariance = np.array([self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0,2],
        #                                          self.Pk[1, 0], self.Pk[1, 1], 0, 0, 0, self.Pk[1,2],
        #                                          0, 0, 0, 0, 0, 0,
        #                                          0, 0, 0, 0, 0, 0,
        #                                          0, 0, 0, 0, 0, 0,
        #                                          self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2,2]])
                                        
        self.state_pub.publish(state)
    
    def publish_features(self, event):
        ''' Publishes the poses and uncertainties of markers detected '''
        
        if self.nf > 0:
            features = np.split(self.xk[self.xBpose_dim:],self.nf)   # split features from state
            aruco_array = ArucoWithCovarianceArray()
            aruco_array.header.stamp = rospy.Time.now()
            aruco_array.header.frame_id = "odom"
            for i in range(len(features)):
                feat = features[i]
                feat_id = self.feature_ids[i]
                pose_msg = PoseWithCovariance()
                pose_msg.pose.position.x = feat[0]
                pose_msg.pose.position.y = feat[1]
                Pxx = self.P[self.xBpose_dim+(2*i),self.xBpose_dim+(2*i)]
                Pyy = self.P[self.xBpose_dim+(2*i+1),self.xBpose_dim+(2*i+1)]
                cov = [
                    Pxx, 0,   0, 0, 0, 0,
                    0,   Pyy, 0, 0, 0, 0,
                    0,   0,   0, 0, 0, 0,
                    0,   0,   0, 0, 0, 0,
                    0,   0,   0, 0, 0, 0,
                    0,   0,   0, 0, 0, 0
                ]
                pose_msg.covariance = cov
                # pose_topic_name = f'/turtlebot/kobuki/SLAM/markers/pose_cov_{feat_id}'
                
                # if feat_id not in self.pose_publishers:
                    # print('publishing feature')
                    # self.pose_publishers[feat_id] = rospy.Publisher(pose_topic_name, PoseWithCovariance, queue_size=1)
                
                marker_odom_msg = Odometry()
                marker_odom_msg.header.stamp = rospy.Time.now()
                marker_odom_msg.header.frame_id = "odom"
                marker_odom_msg.pose.pose.position.x = feat[0]
                marker_odom_msg.pose.pose.position.y = feat[1]
                marker_odom_msg.pose.covariance = cov
                
                self.pose_publishers[feat_id].publish(marker_odom_msg)
                # print('Publisher>>> publishers:',self.pose_publishers, self.pose_publishers[feat_id])

                marker = ArucoWithCovariance()
                marker.id = feat_id
                marker.pose = pose_msg
                
                
                # print('Publisher >>>id:',feat_id,'Pxx:',Pxx,'Pyy:',Pyy)
                aruco_array.markers.append(marker)
                self.markers_pub.publish(aruco_array)

    
    def joint_state_callback_sim(self, msg):
        
        if msg.name[0] == self.left_wheel_name:         # left wheel
            self.left_wheel_velocity = msg.velocity[0]  # [rad/s] same vel shown in /turtlebot/kobuki/commands/wheel_velocities
            self.left_wheel_received = True
            return
        elif msg.name[0] == self.right_wheel_name:      # right wheel
            self.right_wheel_velocity = msg.velocity[0] # [rad/s]

            if (self.left_wheel_received):              # both wheels received
                if (not self.first_received):           # first time -> for updating dt and do differentials
                    self.first_received = True
                    self.last_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
                    return
                
                # Calculate dt [seconds]
                current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
                dt = (current_time - self.last_time).to_sec()
                self.last_time = current_time

                # Prediction
                # Integrate position
                wheel_velocities = np.array([self.left_wheel_velocity, self.right_wheel_velocity])
                
                ###############
                # [self.x, self.y, self.th] = self.f(dt, wheel_velocities)
                # self.xk = np.array([self.x, self.y, self.th]).reshape(3,1)
                # # Propagate covariance
                # self.P = self.Jfx(dt, wheel_velocities) @ self.P @ self.Jfx(dt, wheel_velocities).T  + self.Jfw(dt) @ np.diag(self.wheel_vel_noise) @ self.Jfw(dt).T
                ###############
                self.xk, self.P = self.Prediction(self.xk, self.P, np.diag(self.wheel_vel_noise), dt, wheel_velocities)

                # Calculate velocity
                lin_vel = wheel_velocities * self.wheel_radius
                v = (lin_vel[0] + lin_vel[1]) / 2.0
                w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
                
                # Predict state
                # self.eta_k, self.Pk = self.prediction(dt, wheel_velocities)

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

                # TODO: consider only updating state and publishing in a single place
                # Include velocity v and w in the publisher
                self.odom_pub.publish(odom)

                self.tf_br.sendTransform((self.x, self.y, 0.0), q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)

    def joint_state_callback_physical(self, msg):
        
        # print('joint_state_callback>>> msg:',msg, '\nvels:',msg.velocity)
        
        # if msg.name[0] == self.left_wheel_name:         # left wheel
        #     self.left_wheel_velocity = msg.velocity[0]  # [rad/s] same vel shown in /turtlebot/kobuki/commands/wheel_velocities
        #     self.left_wheel_received = True
        #     return
        # elif msg.name[0] == self.right_wheel_name:      # right wheel
        #     self.right_wheel_velocity = msg.velocity[0] # [rad/s]

        self.left_wheel_velocity = msg.velocity[0]
        self.right_wheel_velocity = msg.velocity[1]

        if (not self.first_received):           # first time -> for updating dt and do differentials
            self.first_received = True
            self.last_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            return
        
        # Calculate dt [seconds]
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Prediction
        # Integrate position
        wheel_velocities = np.array([self.left_wheel_velocity, self.right_wheel_velocity])
        
        ###############
        # [self.x, self.y, self.th] = self.f(dt, wheel_velocities)
        # self.xk = np.array([self.x, self.y, self.th]).reshape(3,1)
        # # Propagate covariance
        # self.P = self.Jfx(dt, wheel_velocities) @ self.P @ self.Jfx(dt, wheel_velocities).T  + self.Jfw(dt) @ np.diag(self.wheel_vel_noise) @ self.Jfw(dt).T
        ###############
        self.xk, self.P = self.Prediction(self.xk, self.P, np.diag(self.wheel_vel_noise), dt, wheel_velocities)

        # Calculate velocity
        lin_vel = wheel_velocities * self.wheel_radius
        v = (lin_vel[0] + lin_vel[1]) / 2.0
        w = (lin_vel[0] - lin_vel[1]) / self.wheel_base_distance
        
        # Predict state
        # self.eta_k, self.Pk = self.prediction(dt, wheel_velocities)

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

        # TODO: consider only updating state and publishing in a single place
        # Include velocity v and w in the publisher
        self.odom_pub.publish(odom)

        self.tf_br.sendTransform((self.x, self.y, 0.0), q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)

    def Prediction(self, xk_1, Pk_1, Qk, dt, wheel_velocities):
        ''' Motion model prediction with extended state vector for SLAM '''

        # print('Prediction Input>>> xk_1:',xk_1,'\nPk_1:',Pk_1.shape,'\nQk:',Qk,'\ndt:',dt,'\nwheel_velocities:',wheel_velocities)
        # state
        # xBk_bar = self.f(xk_1[0:self.xBpose_dim], uk)       # robot estimation -> EKF_3DOFDifferentialDriveInputDisplacement.f
        [self.x, self.y, self.th] = self.f(dt, wheel_velocities)
        xBk_bar = np.array([self.x, self.y, self.th]).reshape(3,1)
        xk_bar = xk_1                                       # copy state for maintaining features
        xk_bar[0:self.xBpose_dim] = xBk_bar                 # xk bar - only update robot pose

# self.P = self.Jfx(dt, wheel_velocities) @ self.P @ self.Jfx(dt, wheel_velocities).T  + self.Jfw(dt) @ np.diag(self.wheel_vel_noise) @ self.Jfw(dt).T


        # covariance
        F1k = np.eye(xk_1.size)                     # F1k
        # Ak = self.Jfx(xk_1[0:self.xBpose_dim], uk)  # wrt state [3x3] -> EKF_3DOFDifferentialDriveInputDisplacement.Jfx
        Ak = self.Jfx(dt, wheel_velocities)
        F1k[0:Ak.shape[0],0:Ak.shape[1]] = Ak
        F2k = np.zeros((xk_1.shape[0],self.xB_dim)) # F2k
        # Wk = self.Jfw(xk_1[0:self.xBpose_dim])      # wrt noise [3x2]-> EKF_3DOFDifferentialDriveInputDisplacement.Jfw
        Wk = self.Jfw(dt)
        F2k[0:Wk.shape[0],0:Wk.shape[1]] = Wk
        Pk_bar = F1k@Pk_1@F1k.T + F2k@Qk@F2k.T      # Pk bar

        # print('Prediction Output>>> xk_1:',xk_1,'\nPk_1:',Pk_1.shape)

        return xk_bar, Pk_bar

    def imu_callback(self, msg):
        ''' Obtain and Update the orientation of robot XY plane delivered from the IMU '''
        # print('IMU Received')
        q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # yaw = wrap_angle(yaw -np.pi/2) # to align with odom frame
        # self.yaw_Rm = cov[-1]    # alternatively: from 3x3 mat (roll,pitch,yaw) -> element at [2,2]
        self.imu_c += 1
        if self.imu_c % self.imu_update_freq == 0:
            # self.eta_k, self.Pk = self.Update_imu(yaw, self.yaw_Rm, self.eta_k.copy(), self.Pk.copy(), self.Hm, self.Vm)
            # xk_bar = np.array([self.x, self.y, self.th]).reshape(3,1)
            # print('imu_callback>>> xk_bar:',xk_bar)
            Hm = np.hstack((self.Hm,np.zeros((1,self.nf*self.zfi_dim))))
            self.Update(yaw, self.yaw_Rm, self.xk, self.P, self.Hm, self.Vm, 'measurements')
            # print('Updated from IMU')

    def markers_callback(self, msg):
        '''Obtain and Update using aruco markers'''
        self.aruco_c += 1
        if self.aruco_c % self.aruco_update_freq == 0:
            zf = []
            ids = []
            for m in msg.markers:
                x = m.pose.position.x    # points right
                y = m.pose.position.y    # points down (constant - can be ignored)
                z = m.pose.position.z    # points inwards the marker
                BxF = np.array([z,x]).reshape((2,1))
                zf.append(F.CartesianFeature(BxF))
                ids.append(m.id)
            self.nzf = len(zf)
            self.H = self.DataAssociation(self.xk, self.P, ids, zf, self.xy_Rk)
            zp, Rp, Hp, Vp, znp, Rnp, ids_p, ids_np = self.SplitFeatures(zf, self.xy_Rk, self.H, ids)
            self.Update(zp, Rp, self.xk, self.P, Hp, Vp, 'features')
            self.xk, self.P = self.AddNewFeatures(self.xk, self.P, znp, Rnp, ids_np)
        # print(self.aruco_c)
        

        print('xk:',self.xk, '\nfeatures:',self.nf,'\nfeature_ids:',self.feature_ids)
    
    def DataAssociation(self, xk_bar, Pk_bar, ids, zf, Rf):
        ''' Performs the Hypothesis for feature pairing '''

        # expected feature observations and covariance
        # hf = []
        # Phf = []
        # for i in range(self.nf):
        #     hf.append(self.hfj(xk_bar,i))
        #     J_hfjx = self.Jhfjx(xk_bar,i)
        #     Phf.append(J_hfjx @ Pk_bar @ J_hfjx.T)

        # print('DataAssocaition Inputs>>> xk_bar:',xk_bar,'\nids:',ids,'\nzf:',zf,'\nself.ids:',self.feature_ids)
        # features = np.split(xk_bar[self.xBpose_dim:],self.nf)   # split features from state
        # print('features:',features)
        Hp = []
        for i in range(len(ids)):
            if ids[i] in self.feature_ids:
                Fj = self.feature_ids.index(ids[i])
                # feature = features[Fj]
                hf = self.hfj(xk_bar, Fj)       # Q. should we use feature instead?
                Jhfjx = self.Jhfjx(xk_bar, Fj)
                Phf = Jhfjx@Pk_bar@Jhfjx.T
                D2ij = self.SquaredMahalanobisDistance(hf, Phf, zf[i], Rf)
                if self.IndividualCompatibility(D2ij, self.xF_dim, self.alpha):
                    Hp.append(Fj)
                else:
                    Hp.append("spurious") # Q. can we put 'rejected' instead and filter to avoid making it a new candidate feature
                    # Hp.append(None)
            else:
                Hp.append(None)
        # Hypothesis
        # Hp = self.ICNN(hf, Phf, zf, Rf, self.xF_dim)
        # print('DataAssociation Outputs>>> Hp:',Hp)
        return Hp
    
    def ICNN(self, hf, Phf, zf, Rf, dim):
        '''
        Individual Compatibility Nearest Neighbor (ICNN) data association algorithm. 
        Returns a pairing hypothesis that associates each feature observation with the expected feature observation
        that minimizes the Mahalanobis distance.
        '''

        Hp = []
        for j in range(self.nzf):        # zFj -> for every feature sensed
            nearest = None
            D2min = np.inf
            for i in range(self.nf):    # xFi -> for every feature in the Map
                D2ij = self.SquaredMahalanobisDistance(hf[i], Phf[i], zf[j], Rf)
                if self.IndividualCompatibility(D2ij, dim, self.alpha) and D2ij < D2min:
                    nearest = i
                    D2min = D2ij
            Hp.append(nearest)
        return Hp
    
    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        ''' Squared Mahalanobis distance between the expected feature observation and the feature observation '''

        vij = zfi - hfj     # error computation
        Sij = Rfi + Pfj     # total covariance
        D2ij = vij.T @ np.linalg.pinv(Sij) @ vij    # Mahalanobis distance
        return D2ij

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        '''
        Computes the individual compatibility test for the squared Mahalanobis distance.
        Chi-Square distribution with 'dof' degrees of freedom and a significance level 'alpha'.
        '''
        
        thr = scipy.stats.chi2.ppf(alpha, dof)
        isCompatible = D2_ij <= thr
        return isCompatible
    
    def SplitFeatures(self, zf, Rf, H, ids):
        ''' Split paired and non-paired features from Hypothesis '''
        
        # print('SplitFeatures Inputs>>> zf:',zf,'\nH:',H,'\nids:',ids)
        zp = np.zeros((0,1))                  # empty [0x1]
        Hp = np.zeros((0,self.xBpose_dim + self.nf*self.zfi_dim))   # empty [0xD] D -> size of xk: pose + features
        Rp = np.zeros((0,0))                  # empty [0x0]
        Vp = np.zeros((0,0))                  # empty [0x0]
        znp = np.zeros((0,1))                 # empty [0x1]
        Rnp = np.zeros((0,0))                 # empty [0x0]
        ids_p = []
        ids_np = []

        for i in range(len(H)):     # for all associations
            Fj = H[i]
            if Fj != "spurious":
                if Fj is not None:   # paired
                    zp = np.vstack((zp,zf[i]))
                    Rp = scipy.linalg.block_diag(Rp,Rf)
                    Hp = np.vstack((Hp,self.Jhfjx(self.xk,Fj)))
                    Vp = scipy.linalg.block_diag(Vp,np.eye(self.zfi_dim))
                    ids_p.append(ids[i])
                else:     # not paired
                    znp = np.vstack((znp,zf[i]))
                    Rnp = scipy.linalg.block_diag(Rnp,Rf)
                    ids_np.append(ids[i])
        # print('SplitFeatures Outputs>>> zp:',zp,'\nznp:',znp,'\nids_p:',ids_p,'\nids_np:',ids_np,'\nHp:',Hp)
        return zp, Rp, Hp, Vp, znp, Rnp, ids_p, ids_np
    
    def AddNewFeatures(self, xk, Pk, znp, Rnp, ids_np):
        ''' Extends the state vector to have robot pose and map '''
        
        # print('AddNewFeatures Inputs>>> xk:',xk,'\nznp:',znp,'\nids_np:',ids_np)
        
        if znp.size == 0:
            return xk, Pk
        n_newf = int(len(znp)/self.zfi_dim)         # number of unpaired features
        self.nf += n_newf                           # num features to add in vector state
        # NxB = self.GetRobotPose(xk)                 # robot pose
        NxB = P.Pose3D(xk[0:self.xBpose_dim])
        features = np.split(znp,n_newf)             # split features by pairs
        for i in range(len(features)):              # for each feature to be added
            feat = features[i]
            # state
            BxFj = F.CartesianFeature(feat)                       # convert to CartesianFeatures - depends on type of feature
            # MFobj = MF.MapFeature(BxFj)
            xk_plus = np.vstack((xk,self.g(NxB,BxFj)))          # xk+ - stack at the bottom
            # covariance
            G1k = np.zeros((xk.size+self.xF_dim,xk.size))       # G1k
            G1k[0:xk.size,0:xk.size] = np.eye(xk.size)
            J1box = BxFj.J_1boxplus(NxB)
            G1k[-self.xF_dim:,0:J1box.shape[1]]= J1box
            G2k = np.zeros((xk_plus.size,self.xF_dim))          # G2k
            J2box = BxFj.J_2boxplus(NxB)
            G2k[-2:] = J2box
            Rk = Rnp[i*2:i*2+2,i*2:i*2+2]                       # Rk
            Pk_plus = G1k@Pk@G1k.T + G2k@Rk@G2k.T               # Pk+
            
            # update iteration
            self.feature_ids.append(ids_np[i])
            xk = xk_plus
            Pk = Pk_plus

        # print('AddNewFeatures Outputs>>> xk_plus:',xk_plus,'\nself.feature_ids:',self.feature_ids)
        return xk_plus, Pk_plus
    
    def g(self, xk, BxFj):  # xBp [+] (BxFj + vk)
        ''' Inverse observation model '''

        NxB = xk # self.GetRobotPose(xk)
        # BxFj = self.M[BxFj]
        # NxFj = self.o2s(BxFj).boxplus(NxB)
        NxFj = BxFj.boxplus(NxB)
        return NxFj

    def hfj(self, xk_bar, Fj):  # Observation function for zf_i and x_Fj
        ''' Direct observation model for a single feature observation '''
        
        # s2o( (-)NxB [+] NxF)
        NxB = P.Pose3D(xk_bar[0:self.xBpose_dim])               # robot pose
        features = np.split(xk_bar[self.xBpose_dim:],self.nf)   # split features from state

        # print('hfj>>> xk_bar:',xk_bar,'features:',features,'features[Fj]:',features[Fj],'Fj:',Fj)
        NxF = F.CartesianFeature(features[Fj])                  # selected feature as CartesianFeature
        # MFobj = MF.MapFeature(NxF)
        # _hfj = MFobj.s2o(NxF.boxplus(NxB.ominus()))                # compute observation
        _hfj = NxF.boxplus(NxB.ominus())
        return _hfj

    def Jhfjx(self, xk_bar, Fj):  # Observation function for zf_i and x_Fj
        ''' Jacobian of the single feature direct observation model w.r.t. the state '''

        NxB = P.Pose3D(xk_bar[0:self.xBpose_dim])                 # robot pose
        features = np.split(xk_bar[self.xBpose_dim:],self.nf)   # split features from state
        NxF = F.CartesianFeature(features[Fj])                # selected feature as CartesianFeature

        # J_NxB = MF.J_s2o(NxF.boxplus(NxB.ominus())) @ NxF.J_1boxplus(NxB.ominus()) @ NxB.J_ominus()
        # J_NxF = MF.J_s2o(NxF.boxplus(NxB.ominus())) @ NxF.J_2boxplus(NxB)
        J_NxB = NxF.J_1boxplus(NxB.ominus()) @ NxB.J_ominus()
        J_NxF = NxF.J_2boxplus(NxB)
        # print('Jhfjx>>> J_NxB:',J_NxB)
        # print('Jhfjx>>> J_NxF:',J_NxF)
        Jp = np.zeros((self.xF_dim,xk_bar.size)) # position
        Jp[:,0:self.xBpose_dim] = J_NxB
        idx = self.xBpose_dim + Fj*2
        Jp[:, idx:idx+self.xF_dim] = J_NxF
        # Jnp = np.zeros((self.xF_dim,self.xBpose_dim))     # not position -> velocities
        # J = np.hstack((Jp, Jnp))
        return Jp
    
    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, sensor):
        ''' Perform Kalman equation for Update '''
        
        # print(f"Update Inputs {sensor}>>> zk:",zk,"\nRk:",Rk.shape,"\nxk_bar:",xk_bar,"\nPk_bar:",Pk_bar.shape,"\nHk:",Hk.shape,"\nVk:",Vk.shape)
        # Kk = Pk_bar@Hk.T @ np.linalg.pinv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T) # Kalman Gain [3x1] -> pseudoinverse works for 1x1 and bigger matrices
        # xk = xk_bar + Kk@np.atleast_2d(zk - self.h(xk_bar)).reshape(-1,1) # updated state [3x1] -> atleast2d ensures compatibility with 1x1 and bigger matrices
        
        # print(Kk.reshape(3,1) @ wrap_angle(zk - self.h(xk_bar)).reshape(-1,1) )

        # eta_k = xk_bar + Kk.reshape(3,1) @ wrap_angle(zk - self.h(xk_bar)).reshape(-1,1) # updated state [3x1] -> atleast2d ensures compatibility with 1x1 and bigger matrices
        if sensor == 'measurements':
            Hk = np.hstack((Hk,np.zeros((1,self.nf*self.zfi_dim))))
            Kk = Pk_bar@Hk.T @ np.linalg.pinv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T) # Kalman Gain [3x1] -> pseudoinverse works for 1x1 and bigger matrices
            diff = zk - self.hm(xk_bar)
            xk = xk_bar + Kk@np.atleast_2d(wrap_angle(diff)).reshape(-1,1)
        elif sensor == 'features':
            Kk = Pk_bar@Hk.T @ np.linalg.pinv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T) # Kalman Gain [3x1] -> pseudoinverse works for 1x1 and bigger matrices
            diff = zk - self.hf(xk_bar)
            xk = xk_bar + Kk@np.atleast_2d(diff).reshape(-1,1)
        # print(f'Update {sensor}>>> zk - self.h(xk_bar){diff}')

        I = np.eye(Pk_bar.shape[0])
        Pk = (I - Kk@Hk) @ Pk_bar @ (I - Kk@Hk).T
        
        # print(f'Update Mid {sensor}>>> xk:',xk,'\nPk:',Pk.shape)
        # return eta_k.reshape(-1,1), Pk
        self.x = xk[0][0]
        self.y = xk[1][0]
        self.th = xk[2][0]
        self.xk = xk
        self.P = Pk
        # print(f"Update Output {sensor}>>> self.xk:",self.xk, 'self.P:',self.P.shape,'x,y,th:',self.x,self.y,self.th)

    
    def hm(self, xk_bar):
        ''' Actual state yaw of the robot '''

        return xk_bar[2]
    
    def hf(self, xk):  # Observation function for all zf observations
        '''
        This is the direct observation model, implementing the feature observation equation for the data
        association hypothesis.
        '''

        #### TODO: consider simplifying hf and hm into a global h(sensor) in Update to make it cleaner
        #### and handle each case in the h function that calls hm and hf respectively
        # print('hf Inputs>>> xk:',xk,'\nzf:',zf)
        _hf = []
        for i in range(len(self.H)):
            Fj = self.H[i]
            # print('hf >>> self.H:',self.H,'Fj:',Fj)
            if Fj is not None and Fj != "spurious":
                _hf.append(self.hfj(xk, Fj))

        # print('hf>>> _hf:',_hf)
        # print(len(_hf))
        if len(_hf) == 0:
            return np.zeros((0,0))      #################### TODO: not needed
        else:
            return np.vstack(_hf)


if __name__ == '__main__':

    rospy.init_node("FEKFSLAM")
    print('Debug info', flush=True)

    robot = DifferentialDrive()

    rospy.spin()

    





