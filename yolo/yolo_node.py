#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
功能：图像检测
耦合点：
    1、只有当img的state置为RUNNING时，才能正常获取到图片
    2、会对detect进行写入，传递给处理线程
时间：2025/4/3,15:07
作者：kkking789
"""

import torch
import copy
import yolo.yolo5_detect as yolo5_detect
import cv2
import time
import rospy
from cv_bridge import CvBridge

from YoloV5Detector.V5Detector import Detector
from sensor.msg import Todetect
from yolo.msg import Detected
from sensor_msgs.msg import Imu, NavSatFix, Image

weights = './yolo/weights/best.pt'
yaml = './yolo/yaml/data.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class YoloNode:
    def __init__(self):
        self.todetectsub = rospy.Subscriber("/todetect", Todetect, self.todetectcallback, queue_size=2)
        self.detectedpub = rospy.Publisher("/detected", Detected, queue_size=2)

        # 网络初始化
        self.det = Detector(weights, yaml, device=device)
        self.yolo5_module = yolo5_detect.yolo5_arr(self.det, yaml=yaml)
        self.pre_time = time.time()

        self.todetect = None
        self.imgmsg = Image()
        self.bridge = CvBridge()
        self.rpy = []
        self.pos = []
        self.lock = False

    def todetectcallback(self, tdmsg):
        if not self.lock:
            self.imgmsg = copy.deepcopy(tdmsg.img)
            self.todetect = self.bridge.imgmsg_to_cv2(tdmsg.img)
            self.pos = tdmsg.pose
            self.rpy = tdmsg.rpy

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.lock = True
            if self.todetect is None:
                rate.sleep()
                continue
            fps = 1 / (time.time() - pre_time)  # 计算实时帧数
            pre_time = time.time()
            # 数据预处理
            todetect = cv2.cvtColor(self.todetect, cv2.COLOR_BGR2RGB)
            todetect = cv2.convertScaleAbs(todetect, alpha=1.2, beta=0)
            # 推理
            dst_mat, det_res = self.yolo5_module.detecting(todetect, img_size=(640, 640))
            # cv2.putText(dst_mat, f"fps:{fps}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)
            # cv2.imshow("dst", dst_mat)
            # cv2.waitKey(1)
            detected = Detected()
            detected.pred_boxes = det_res
            detected.img = self.imgmsg
            detected.pose = self.pos
            detected.rpy = self.rpy

            self.detectedpub.publish(detected)
            self.lock = False

            rate.sleep()

if __name__ == '__main__':
    node = YoloNode()
    node.run()
