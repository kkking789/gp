"""
功能：各类数据处理，将最终目标点存储并发送至Track
耦合点：从Yolo中获取追踪点数据，向Track发送目标点数据
时间：2025/4/4，10：41
作者：kkking789
"""
import math
import copy

from share import *
from math import sqrt, cos, sin, asin, atan
import cv2

img0: DetectClass = None
Ki, mapx, mapy, stereo = None, None, None, None


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


# 极线矫正
def polarimetric_correct(img1: DetectClass):
    mid_yaw = (img0.rpy[2] + img1.rpy[2]) / 2
    mid_pitch = (img0.rpy[1] + img1.rpy[1]) / 2
    mid_roll = (img0.rpy[0] + img1.rpy[0]) / 2

    r2 = calculateR(mid_yaw, mid_pitch, mid_roll)
    r1i = np.linalg.inv(calculateR(img1.rpy[2], img1.rpy[1], img1.rpy[0]))
    pr1i = np.linalg.inv(calculateR(img0.rpy[2], img0.rpy[1], img0.rpy[0]))

    pm = K @ r2 @ pr1i @ Ki
    m = K @ r2 @ r1i @ Ki

    img_ = cv2.remap(img1.img, mapx, mapy, cv2.INTER_LINEAR)
    pimg = cv2.remap(img0.img, mapx, mapy, cv2.INTER_LINEAR)

    pcimg = cv2.warpPerspective(pimg, pm, (W, H_))
    cimg = cv2.warpPerspective(img_, m, (W, H_))

    # cv2.imshow("pcimg", pcimg)
    # cv2.imshow("cimg", cimg)

    return pcimg, cimg


# 计算深度图，单位分米
def deep(img1: ImgClass):
    global img0
    if img0 is None:
        img0 = img1
    depth_map = None
    distance = sqrt((img1.gps.x - img0.gps.x) ** 2 + (img1.gps.y - img0.gps.y) ** 2)
    if distance > 0.1:
        pcimg, cimg = polarimetric_correct(img1)
        disparity = stereo.compute(pcimg, cimg).astype(np.float32) / 16.0
        depth_map = (F * distance * 10) / disparity
        depth_map[disparity == 0] = 0  # 设置无效区域的深度为0
        depth_map[depth_map > 50] = 0  # 对过远点归零
        img0 = img1
    return depth_map


# x，y为目标点像素坐标，坐标原点为左上角
def targetpoint(x, y, depth_map):
    distance = 0
    radius = 2
    while distance == 0:
        y_min, y_max = max(0, y - radius), min(H_, y + radius)
        x_min, x_max = max(0, x - radius), min(W, x + radius)
        radius += 1
        neighbor = depth_map[y_min:y_max, x_min:x_max]
        nz = neighbor[neighbor != 0]
        if len(nz) == 0:
            continue
        distance = np.mean(nz)

    yaw = math.degrees(atan((x - K[0][2]) / K[0][0]))

    return yaw, distance


def brain():
    global Ki, mapx, mapy, stereo, img0, npos
    Ki = np.linalg.inv(K)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (W, H_), 5)
    point = []
    # SGBM参数设置
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
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

    while True:
        img1 = None
        depth_map = None
        with detectLock:
            if detect.state == RUNNING_STATE:
                img1 = copy.deepcopy(detect)
        if img1 is not None:
            depth_map = deep(img1)
        if depth_map is not None:
            pixelx = (img0.pred_boxes[0] + img1.pred_boxes[0] + img0.pred_boxes[2] + img1.pred_boxes[2]) / 4
            pixely = (img0.pred_boxes[1] + img1.pred_boxes[1] + img0.pred_boxes[3] + img1.pred_boxes[3]) / 4
            yaw, distance = targetpoint(pixelx, pixely, depth_map)

            mid_x = (img0.gps.x + img1.gps.x) / 2
            mid_y = (img0.gps.y + img1.gps.y) / 2
            pointx = mid_x + distance * sin(yaw)
            pointy = mid_y + distance * cos(yaw)
            point.append((pointx, pointy))
        if len(point) > 0:
            with nposLock:
                if npos.state == INITIAL_STATE or npos.feedbackstate == FINISH_STATE:
                    pos = point.pop()
                    l = sqrt((npos.targetx-pos[0]) ** 2 + (npos.targety-pos[1])**2)
                    if l >= 5:
                        npos.state = RUNNING_STATE
                        npos.targetx = pos[0]
                        npos.targety = pos[1]


class BrainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        brain()
