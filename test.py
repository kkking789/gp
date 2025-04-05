"""
功能：接收摄像头与传感器数据
耦合点：无，纯向外发送数据
时间：2025/4/4，14:04
作者：kkking789
"""

import math
import cv2
import copy
import rospy
from std_msgs.msg import Imu, NavSatFix

from pyproj import Transformer

gpscopy, rpy = None, None


def wgs2utm(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651")
    east, north = transformer.transform(lat, lon)
    return east, north


def q2e(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = math.asin(2 * (q0 * q2 - q3 * q1))
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    if yaw < 0:
        yaw += 2 * math.pi
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def imucallback(imumsg):
    global rpy, imu
    print(imumsg.orientation)


def gpscallback(gpsmsg):
    global gpscopy,gps
    print(gpsmsg)


def getsensor():
    global W, H_, img
    rospy.init_node("nanosensor")
    imusub = rospy.Subscriber("/imu/data_raw", Imu, imucallback, queue_size=10)
    gpsub = rospy.Subscriber("/senor/m_fix", NavSatFix, gpscallback, queue_size=10)

    cap = cv2.VideoCapture(2)
    W, H_ = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        rospy.spinOnce()






