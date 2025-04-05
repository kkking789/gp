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
from sensor_msgs.msg import Imu, NavSatFix

from pyproj import Transformer
from share import *

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
    with imuLock:
        imu.state = RUNNING_STATE
        imu.orientation = copy.copy([imumsg.orientation.x, imumsg.orientation.y, imumsg.orientation.z, imumsg.orientation.w])
        imu.angular_velocity = copy.copy([imumsg.angular_velocity.x, imumsg.angular_velocity.y, imumsg.angular_velocity.z, imumsg])
        imu.linear_acceleration = copy.copy([imumsg.linear_acceleration.x, imumsg.linear_acceleration.y,
                                   imumsg.linear_acceleration.z])
        imu.rpy = q2e(imu.orientation)
        rpy = imu.rpy


def gpscallback(gpsmsg):
    global gpscopy,gps
    with gpsLock:
        gps.state = RUNNING_STATE
        gps.latitude = copy.copy(gpsmsg.latitude)
        gps.longitude = copy.copy(gpsmsg.longitude)
        gps.x, gps.y = wgs2utm(gps.latitude, gps.longitude)
        gpscopy = gps


def getsensor():
    global W, H_, img
    rospy.init_node("nanosensor")
    imusub = rospy.Subscriber("/senor/m_imu", Imu, imucallback, queue_size=10)
    gpsub = rospy.Subscriber("/senor/m_fix", NavSatFix, gpscallback, queue_size=10)

    cap = cv2.VideoCapture(0)
    W, H_ = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        ret, frame = cap.read()
        if ret and rpy is not None and gpscopy is not None:
            with imgLock:
                img.state = RUNNING_STATE
                img.img = copy.deepcopy(frame)
                img.rpy = copy.copy(rpy)
                img.gps = copy.copy(gpscopy)


class SensorThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        getsensor()



