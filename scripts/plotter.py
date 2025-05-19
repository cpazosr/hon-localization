#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from localization.msg import ArucoWithCovarianceArray
from matplotlib.patches import Ellipse
import message_filters
import threading

class SLAMPlotter:
    def __init__(self):
        rospy.init_node("slam_plotter")
        self.odom_sub = message_filters.Subscriber("/turtlebot/kobuki/SLAM/EKF_odom", Odometry)
        self.markers_sub = message_filters.Subscriber("/turtlebot/kobuki/SLAM/markers", ArucoWithCovarianceArray)
        self.gt_sub = message_filters.Subscriber("/turtlebot/kobuki/odom_ground_truth",Odometry)

        # Robot
        self.lock = threading.Lock()
        self.robot_pos   = None            # (x,y)
        self.robot_cov   = None            # 3×3
        self.gt_pos      = None            # (x,y)
        self.markers     = {}              # {id : (x,y,cov2x2)}

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.markers_sub, self.gt_sub],
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
            # Get robot position
            rx = odom_msg.pose.pose.position.x
            ry = odom_msg.pose.pose.position.y
            self.robot_pos = (rx, ry)
            c = odom_msg.pose.covariance
            self.robot_cov = np.array([[c[0], c[1], c[2]],
                                       [c[6], c[7], c[8]],
                                       [c[12],c[13],c[14]]])
            # print(f'sync_callback>>> rx:{rx},ry:{ry},cov:{self.robot_cov}')

            # Ground truth
            gtx = gt_msg.pose.pose.position.x
            gty = gt_msg.pose.pose.position.y
            self.gt_pos = (gtx, gty)

            # Markers
            self.markers.clear()
            for m in marker_msg.markers:
                x,y = m.pose.pose.position.x, m.pose.pose.position.y
                c   = m.pose.covariance
                cov = np.array([[c[0], c[1]],[c[6], c[7]]])
                self.markers[m.id] = (x, y, cov)
    
    def redraw(self, _event):
        with self.lock:
            if self.robot_pos is None:   # nothing yet
                return
            # Robot
            rx, ry         = self.robot_pos
            r_cov          = self.robot_cov[0:2,0:2]   # only x-y plotting (2D)
            markers_dict   = self.markers.copy()
            # Ground truth
            gtx, gty = self.gt_pos

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title("Robot & ArUco landmarks with 1-σ ellipses")
        self.ax.grid(True)

        # Robot
        self.ax.plot(rx, ry, 'bo', label="Robot")
        self.draw_ellipse(rx, ry, r_cov, edge='blue')

        # Ground truth
        self.ax.plot(gtx, gty, 'ro', label="Ground Truth")

        # Markers
        for mid,(mx,my,mcov) in markers_dict.items():
            self.ax.plot(mx, my, 'rx')
            self.ax.text(mx, my, str(mid), color='red', fontsize=8)
            self.draw_ellipse(mx, my, mcov, edge='red')

        self.ax.legend()
        plt.draw()         # GUI work in main thread is safe
        plt.pause(0.001)
    
    @staticmethod
    def draw_ellipse(x, y, cov, edge='k'):
        if cov.shape!=(2,2): return
        vals, vecs = np.linalg.eigh(cov)
        angle      = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        w,h        = 2*np.sqrt(vals)      # 1-σ
        e = Ellipse((x,y), w, h, angle,
                    edgecolor=edge, facecolor='none', lw=1)
        plt.gca().add_patch(e)

    def run(self):
        rospy.spin()

    def odom_callback(self, msg):
        # Extract robot
        self.rx = msg.pose.pose.position.x
        self.ry = msg.pose.pose.position.y
        self.P = np.array(msg.pose.covariance).reshape(6,6)

    def markers_callback(self, msg):
        # Extract markers
        self.markers = msg.markers
    
    def plot(self, event):
        ''' Plotting '''

        # Robot
        self.ax.clear()
        self.ax.plot(self.rx, self.ry, 'bo', label='Robot')
        self.plot_covariance_ellipse(self.rx, self.ry, self.P[0:2, 0:2], color='b')  # robot pose
        
        # Markers
        # markers = np.zeros((0,0))
        for m in self.markers:
            mx = m.pose.pose.position.x
            my = m.pose.pose.position.y
            mP = m.pose.covariance.reshape(6,6)
            mid = m.id
            # markers = np.array(msg.markers).reshape(-1, 2)
            self.ax.plot(mx, my, 'rx', label=f'marker: {mid}')
            self.plot_covariance_ellipse(self.rx, self.ry, self.P[0:1, 0:1], color='b')  # robot pose

            # Covariance matrix
            # N = 3 + 2 * markers.shape[0]
            # P = np.array(msg.covariance).reshape(N, N)

            # Plot uncertainty ellipses
            
            # for i, (lx, ly) in enumerate(markers):
            #     idx = 3 + 2 * i
            #     self.plot_covariance_ellipse(lx, ly, P[idx:idx+2, idx:idx+2], color='r')

        self.ax.set_xlim(self.rx - 5, self.rx + 5)
        self.ax.set_ylim(self.ry - 5, self.ry + 5)
        self.ax.set_aspect('equal')
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def plot_covariance_ellipse(self, x, y, cov, n_std=1.0, color='k'):
        if cov.shape != (2, 2):
            return
        vals, vecs = np.linalg.eigh(cov)
        angle = np.arctan2(vecs[1, 0], vecs[0, 0])
        width, height = 2 * n_std * np.sqrt(vals)
        from matplotlib.patches import Ellipse
        ell = Ellipse(xy=(x, y), width=width, height=height,
                      angle=np.degrees(angle), edgecolor=color,
                      facecolor='none', lw=2)
        self.ax.add_patch(ell)

if __name__ == '__main__':
    # try:
    plotter = SLAMPlotter()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        plotter.redraw(None)
        rate.sleep()
    # except Exception as e:
    #     print(e)
