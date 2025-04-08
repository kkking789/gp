"""
功能：各类数据处理，将最终目标点存储并发送至Track
耦合点：从Yolo中获取追踪点数据，向Track发送目标点数据
时间：2025/4/4，10：41
作者：kkking789
"""
import math
import copy

from math import sqrt, cos, sin, atan
import rospy
import numpy as np
from cv_bridge import CvBridge
from yolo.msg import Detected
from geometry_msgs.msg import Pose2D
import cv2

blockSize = 8
img_channels = 3
K = np.array([[2.817033846637419e+02, 0, 1.559567963466772e+02],
              [0, 2.809657898807247e+02, 1.216932104820509e+02],
              [0, 0, 1]])
D = np.array([[0.015352222035580, -0.092403852639830, 0, 0, 0.108435908257982]])
W = 320
H = 240
F = 10  # 焦距


# 计算旋转矩阵
def calculateR(yaw, pitch, roll):
    rz = np.array([[cos(roll), -sin(roll), 0],
                   [sin(roll), cos(roll), 0],
                   [0, 0, 1]])
    ry = np.array([[cos(yaw), 0, sin(yaw)],
                   [0, 1, 0],
                   [-sin(yaw), 0, cos(yaw)]])
    rx = np.array([[1, 0, 0],
                   [0, cos(pitch), -sin(pitch)],
                   [0, sin(pitch), cos(pitch)]])
    r = rz @ ry @ rx
    return r


# x，y为目标点像素坐标，坐标原点为左上角
def targetpoint(x, y, depth_map):
    distance = 0
    radius = 2
    while distance == 0:
        y_min, y_max = max(0, y - radius), min(H, y + radius)
        x_min, x_max = max(0, x - radius), min(W, x + radius)
        radius += 1
        neighbor = depth_map[y_min:y_max, x_min:x_max]
        nz = neighbor[neighbor != 0]
        if len(nz) == 0:
            continue
        distance = np.mean(nz)

    yaw = atan((x - K[0][2]) / K[0][0])

    return yaw, distance


class BrainNode:
    def __init__(self):
        rospy.init_node("nanobrain")

        self.Ki = np.linalg.inv(K)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), 5)
        # SGBM参数设置
        self.stereo = cv2.StereoSGBM_create(minDisparity=1,
                                            numDisparities=64,
                                            blockSize=blockSize,
                                            P1=8 * img_channels * blockSize * blockSize,
                                            P2=32 * img_channels * blockSize * blockSize,
                                            disp12MaxDiff=-1,
                                            preFilterCap=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=100,
                                            mode=cv2.STEREO_SGBM_MODE_HH)
        self.depth_map = None
        self.detect0 = None
        self.bridge = CvBridge()
        self.point = []
        self.gpoint = Pose2D()
        self.gpoint.x = 0
        self.gpoint.y = 0
        self.detectsub = rospy.Subscriber("detect", Detected, self.detectcallback, queue_size=10)
        self.pointpub = rospy.Publisher("goalpoint", Pose2D, queue_size=2)

    def detectcallback(self, detectmsg):
        self.depth_map = self.deep(detectmsg)
        if self.depth_map is None:
            return
        pixelx = (detectmsg.pred_boxes[0] + detectmsg.pred_boxes[2]) / 2
        pixely = (detectmsg.pred_boxes[1] + detectmsg.pred_boxes[3]) / 2
        yaw, distance = targetpoint(pixelx, pixely, self.depth_map)

        mid_x = (self.detect0.pose.x + detectmsg.pose.x) / 2
        mid_y = (self.detect0.pose.y + detectmsg.pose.y) / 2
        pointx = mid_x + distance * sin(yaw)
        pointy = mid_y + distance * cos(yaw)
        self.point.append((pointx, pointy))

    # 极线矫正
    def polarimetric_correct(self, detect1):
        mid_yaw = (self.detect0.rpy[2] + detect1.rpy[2]) / 2
        mid_pitch = (self.detect0.rpy[1] + detect1.rpy[1]) / 2
        mid_roll = (self.detect0.rpy[0] + detect1.rpy[0]) / 2

        r2 = calculateR(mid_yaw, mid_pitch, mid_roll)
        r1i = np.linalg.inv(calculateR(detect1.rpy[2], detect1.rpy[1], detect1.rpy[0]))
        pr1i = np.linalg.inv(calculateR(self.detect0.rpy[2], self.detect0.rpy[1], self.detect0.rpy[0]))

        pm = K @ r2 @ pr1i @ self.Ki
        m = K @ r2 @ r1i @ self.Ki

        img1 = self.bridge.imgmsg_to_cv2(detect1.img, "bgr8")
        img0 = self.bridge.imgmsg_to_cv2(self.detect0.img, "bgr8")
        img_ = cv2.remap(img1, self.mapx, self.mapy, cv2.INTER_LINEAR)
        pimg = cv2.remap(img0, self.mapx, self.mapy, cv2.INTER_LINEAR)

        pcimg = cv2.warpPerspective(pimg, pm, (W, H))
        cimg = cv2.warpPerspective(img_, m, (W, H))

        return pcimg, cimg

    # 计算深度图，单位分米
    def deep(self, detect1):
        if self.detect0 is None:
            self.detect0 = copy.deepcopy(detect1)
        depth_map = None
        distance = sqrt((detect1.pose.x - self.detect0.pose.x) ** 2 + (detect1.pose.y - self.detect0.pose.y) ** 2)
        if distance > 0.1:
            pcimg, cimg = self.polarimetric_correct(detect1)
            disparity = self.stereo.compute(pcimg, cimg).astype(np.float32) / 16.0
            depth_map = (F * distance * 10) / disparity
            depth_map[disparity == 0] = 0  # 设置无效区域的深度为0
            depth_map[depth_map > 50] = 0  # 对过远点归零
            self.detect0 = copy.deepcopy(detect1)
        return depth_map

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if len(self.point) != 0:
                data = self.point.pop()
                l = sqrt((self.gpoint.x - data[0]) ** 2 + (self.gpoint.y - data[1]) ** 2)
                if l >= 5:
                    self.gpoint.x = data[0]
                    self.gpoint.y = data[1]
                    self.pointpub.publish(self.gpoint)
            rate.sleep()


if __name__ == '__main__':
    node = BrainNode()
    node.run()