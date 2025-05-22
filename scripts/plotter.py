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
import message_filters
import threading
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class SLAMPlotter:
    def __init__(self):
        rospy.init_node("slam_plotter")
        self.odom_sub = message_filters.Subscriber("/turtlebot/kobuki/SLAM/odom", Odometry)
        self.markers_sub = message_filters.Subscriber("/turtlebot/kobuki/SLAM/markers", ArucoWithCovarianceArray)
        self.gt_sub = message_filters.Subscriber("/turtlebot/kobuki/odom_ground_truth",Odometry)

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

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.markers_sub, self.gt_sub],
            # [self.odom_sub, self.gt_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Initialize matplotlib
        self.fig, self.ax = plt.subplots()
        plt.ion()
        print('Init complete...')

    def sync_callback(self, odom_msg, marker_msg, gt_msg):
        with self.lock:
            # Get robot odometry
            rx = odom_msg.pose.pose.position.x
            ry = odom_msg.pose.pose.position.y
            rq = odom_msg.pose.pose.orientation
            _, _, ryaw = euler_from_quaternion([rq.x, rq.y, rq.z, rq.w])
            self.robot_pos = (rx, ry)
            c = odom_msg.pose.covariance
            self.robot_cov = np.array([[c[0], c[1], c[2]],
                                       [c[6], c[7], c[8]],
                                       [c[12],c[13],c[14]]])
            self.robot_xk.append( (rx,ry,ryaw, self.robot_cov) )
            print('sync_callback: robot odometry acquired')

            # Ground truth
            pose_in = PoseStamped()
            pose_in.header = gt_msg.header
            pose_in.pose = gt_msg.pose.pose

            print(f'gt frame_id:{pose_in.header.frame_id}')
            tf = self.tf_buffer.lookup_transform("world_ned",
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
            print('sync_callback: ground truth acquired')

            # Markers
            self.markers.clear()
            for m in marker_msg.markers:
                x,y = m.pose.pose.position.x, m.pose.pose.position.y
                c   = m.pose.covariance
                cov = np.array([[c[0], c[1]],[c[6], c[7]]])
                self.markers[m.id] = (x, y, cov)
            print('sync_callback: markers finished')
            print('sync_callback: finished')
    
    def redraw(self, _event):
        print('redraw')
        with self.lock:
            if self.robot_pos is None:   # nothing yet
                return
            # Robot
            rx, ry = self.robot_pos
            r_cov = self.robot_cov[0:2,0:2]   # only x-y plotting (2D)
            markers_dict = self.markers.copy()
            # Ground truth
            gtx, gty = self.gt_pos

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
        self._plot(axs[0, 0], axs[0, 1], axs[0, 2], x_est,  x_gt,  x_cov,  "x")
        self._plot(axs[1, 0], axs[1, 1], axs[1, 2], y_est,  y_gt,  y_cov,  "y")
        self._plot(axs[2, 0], axs[2, 1], axs[2, 2], th_est, th_gt, th_cov, "θ")

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