#! /usr/bin/env python2
# -*- coding: UTF-8 -*-

"""
功能：接收摄像头与传感器数据
耦合点：无，纯向外发送数据
时间：2025/4/4，14:04
作者：kkking789
"""

import math
import cv2
import rospy
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from sensor.msg import Todetect
from pyproj import Transformer


class SensorNode:
    def __init__(self):
        rospy.init_node("nanosensor")
        self.bridge = CvBridge()
        self.pos = Pose2D()
        self.rpy = [0.0, 0.0, 0.0]

        # 订阅与发布
        self.imusub = rospy.Subscriber("/imu/data_raw", Imu, self.imucallback, queue_size=10)
        self.gpsub = rospy.Subscriber("/sensor/m_fix", NavSatFix, self.gpscallback, queue_size=10)
        self.posepub = rospy.Publisher('/gpspose', Pose2D, queue_size=2)
        self.todetectpub = rospy.Publisher('/todetect', Todetect, queue_size=2)

        # 初始化UTM转换器（动态计算带号）
        self.transformer = None  # 延迟初始化

    def wgs2utm(self, lat, lon):
        if self.transformer is None:
            utm_zone = (int((lon + 180) / 6) % 60) + 1
            epsg_code = f"EPSG:326{utm_zone:02d}"
            self.transformer = Transformer.from_crs("EPSG:4326", epsg_code)
        return self.transformer.transform(lat, lon)

    def imucallback(self, imumsg):
        (roll, pitch, yaw) = euler_from_quaternion([
            imumsg.orientation.x,
            imumsg.orientation.y,
            imumsg.orientation.z,
            imumsg.orientation.w
        ])
        self.rpy = [
            roll,
            pitch,
            yaw % (2 * math.pi)  # 确保yaw在0~360度
        ]

    def gpscallback(self, gpsmsg):
        east, north = self.wgs2utm(gpsmsg.latitude, gpsmsg.longitude)
        self.pos.x = east
        self.pos.y = north
        self.pos.theta = math.radians(self.rpy[2])  # 转换为弧度
        self.posepub.publish(self.pos)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            rospy.logerr("摄像头无法打开")
            return
        rospy.on_shutdown(lambda: cap.release())
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if ret:
                msg = Todetect()
                msg.img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                msg.pose = self.pos
                msg.rpy = self.rpy
                self.todetectpub.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    node = SensorNode()
    node.run()
