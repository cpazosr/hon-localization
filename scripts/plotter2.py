#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from localization.msg import ArucoWithCovarianceArray
from geometry_msgs.msg import PoseStamped
from matplotlib.patches import Ellipse
import threading
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class SLAMPlotter:
    def __init__(self):
        rospy.init_node("slam_plotter")
        # Testing type
        self.physical = False

        self.odom_sub = rospy.Subscriber("/turtlebot/kobuki/SLAM/odom", Odometry, self.odom_callback)
        self.markers_sub = rospy.Subscriber("/turtlebot/kobuki/SLAM/features", ArucoWithCovarianceArray, self.arucos_callback)
        
        if self.physical:
            # Physical
            self.gt_sub = rospy.Subscriber("/turtlebot/kobuki/odom",Odometry, self.gt_callback)
            # self.lidar_slam_sub = rospy.Subscriber("/slam_out_pose",PoseStamped, self.lidar_slam_callback)
        else:
            # Simulation
            self.gt_sub = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth",Odometry,self.gt_callback)

        # Robot
        self.lock = threading.Lock()
        self.robot_pos = None            # (x,y)
        self.robot_cov = None            # 3×3
        self.gt_pos = None               # (x,y)
        self.markers = {}                # {id : (x,y,cov2x2)}

        # Containers for summary plot
        self.robot_xk = []
        self.gt_xk = []

        # Groun truth transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize matplotlib
        self.fig, self.ax = plt.subplots()
        plt.ion()
        print('Init complete...')

    def odom_callback(self, msg):
        # SLAM odometry
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y
        rq = msg.pose.pose.orientation
        _, _, ryaw = euler_from_quaternion([rq.x, rq.y, rq.z, rq.w])
        self.robot_pos = (rx, ry)
        c = msg.pose.covariance
        self.robot_cov = np.array([[c[0], c[1], c[5]],
                                    [c[6], c[7], c[11]],
                                    [c[30],c[31],c[35]]])
        self.robot_xk.append( (rx,ry,ryaw, self.robot_cov) )
    
    def gt_callback(self, msg):
        # Ground truth
        if self.physical:
            gtx = msg.pose.pose.position.x
            gty = -msg.pose.pose.position.y     # negative for visualization
            gtq = msg.pose.pose.orientation
            _, _, gtyaw = euler_from_quaternion([gtq.x, gtq.y, gtq.z, gtq.w])
            self.gt_xk.append((gtx, gty, -gtyaw))
            self.gt_pos = (gtx, gty) 
        else:
            pose_in = PoseStamped()
            pose_in.header = msg.header
            pose_in.pose = msg.pose.pose

            tf = self.tf_buffer.lookup_transform("odom",
                                pose_in.header.frame_id,
                                pose_in.header.stamp,
                                rospy.Duration(0.1))
            pose_out = tf2_geometry_msgs.do_transform_pose(pose_in, tf)
            gtx = pose_out.pose.position.x
            gty = pose_out.pose.position.y
            gtq = pose_out.pose.orientation
            _, _, gtyaw = euler_from_quaternion([gtq.x, gtq.y, gtq.z, gtq.w])
            self.gt_xk.append((gtx, gty, gtyaw))
            self.gt_pos = (gtx, gty)
            
    def lidar_slam_callback(self, msg):
        tf = self.tf_buffer.lookup_transform("odom", #"world_ned",
                            msg.header.frame_id,
                            msg.header.stamp,
                            rospy.Duration(0.1))
        pose_out = tf2_geometry_msgs.do_transform_pose(msg, tf)
        gtx = pose_out.pose.position.x
        gty = pose_out.pose.position.y
        gtq = pose_out.pose.orientation
        _, _, gtyaw = euler_from_quaternion([gtq.x, gtq.y, gtq.z, gtq.w])
        self.gt_xk.append((gtx, gty, gtyaw))
        self.gt_pos = (gtx, gty)

    def arucos_callback(self, msg):
        # Markers
        self.markers.clear()
        for m in msg.markers:
            x,y = m.pose.pose.position.x, m.pose.pose.position.y
            c   = m.pose.covariance
            cov = np.array([[c[0], c[1]],[c[6], c[7]]])
            self.markers[m.id] = (x, y, cov)
    
    
    def redraw(self, _event):
        with self.lock:
            if self.robot_pos is None:   # nothing yet
                return
            # Robot
            rx, ry = self.robot_pos
            r_cov = self.robot_cov[0:2,0:2]   # only x-y plotting (2D)
            # Markers
            markers_dict = self.markers.copy()
            # Ground truth
            gtx, gty = self.gt_pos
            print('data updated')

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Robot pose and Arucos with ellipses")
        self.ax.grid(True)

        # Robot
        self.ax.plot(rx, ry, 'bo', label="Robot")
        self.draw_ellipse(rx, ry, r_cov, edge='blue')

        # Ground truth
        self.ax.plot(gtx, gty, 'g*', label="Ground Truth")

        # Markers
        for mid,(mx,my,mcov) in markers_dict.items():
            self.ax.plot(mx, my, 'rx')
            self.ax.text(mx, my, str(mid), color='red', fontsize=8)
            self.draw_ellipse(mx, my, mcov, edge='red', n_sigma=3)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)
    
    @staticmethod
    def draw_ellipse(x, y, cov, edge='k', n_sigma=1):
        if cov.shape!=(2,2): return             # only 2d plotting
        vals, vecs = np.linalg.eigh(cov)
        angle      = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        w,h        = n_sigma*2*np.sqrt(vals)      # n std_dev
        e = Ellipse((x,y), w, h, angle,
                    edgecolor=edge, facecolor='none', lw=1)
        plt.gca().add_patch(e)
    
    def summary_plot(self):
        # def plot_state_errors(self, unwrap_theta=True, bins=30):
        ''' Plot summary plots comparing robot state vs ground truth over the iterations '''
        # Robot variables - estimated pose
        x_est  = np.array([r[0] for r in self.robot_xk])
        y_est  = np.array([r[1] for r in self.robot_xk])
        th_est = np.array([r[2] for r in self.robot_xk])
        # Robot pose covariance per component
        x_cov  = np.array([r[3][0, 0] for r in self.robot_xk])
        y_cov  = np.array([r[3][1, 1] for r in self.robot_xk])
        th_cov = np.array([r[3][2, 2] for r in self.robot_xk])
        # Ground truth valus
        x_gt, y_gt, th_gt = (np.array(g) for g in zip(*self.gt_xk))

        # if unwrap_theta:
        th_est = np.unwrap(th_est)
        th_gt  = np.unwrap(th_gt)

        # Configure subplots
        fig, axs = plt.subplots(3, 3, figsize=(10, 9))
        idx = np.min([len(x_est), len(x_gt)])
        self._plot(axs[0, 0], axs[0, 1], axs[0, 2], x_est[:idx],  x_gt[:idx],  x_cov[:idx],  "x")
        self._plot(axs[1, 0], axs[1, 1], axs[1, 2], y_est[:idx],  y_gt[:idx],  y_cov[:idx],  "y")
        self._plot(axs[2, 0], axs[2, 1], axs[2, 2], th_est[:idx], th_gt[:idx], th_cov[:idx], "θ")

        plt.tight_layout()
        plt.show(block=True)

        return fig

    def _plot(self, ax_true, ax_err, ax_hist, est, gt, var, label, bins=30):
        ''' Helper plotting function for summary plots '''
        err = est - gt
        std = np.sqrt(var)

        # Ground truth vs robot poses
        ax_true.plot(gt, label="gt", color='green')
        ax_true.plot(est, label="robot", color='blue', alpha=0.7)
        ax_true.set_title(label)
        ax_true.legend(fontsize=8)

        # Covariance traces
        ax_err.plot(err, color='blue', label="error")
        ax_err.plot( 2*std, '--', color='green', label="+2σ")
        ax_err.plot(-2*std, '--', color='green', label="-2σ")
        ax_err.set_title("error")
        ax_err.legend(fontsize=8)

        #   histogram
        ax_hist.hist(err, bins=bins, color='blue', alpha=0.7, density=True)
        ax_hist.axvline(np.mean(err), color='k', linestyle='--')
        ax_hist.set_title("error hist")
        ax_hist.text(0.95, 0.95,
                    f"mean: {np.mean(err):.3f}",
                    ha="right", va="top",
                    transform=ax_hist.transAxes, fontsize=8)

if __name__ == '__main__':
    plotter = SLAMPlotter()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            plotter.redraw(None)
        except Exception as e:
            print(e)
        rate.sleep()
    print('Node terminated after shutdown. Print summary plot')
    plotter.summary_plot()