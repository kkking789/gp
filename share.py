import threading
import numpy as np

INITIAL_STATE = -1
STOP_STATE = 0
RUNNING_STATE = 1
FINISH_STATE = 2
TRACKING_STATE = 3

H = 0.02  # 左舷右舷之间的距离，单位米
K = np.array([[2.817033846637419e+02, 0, 1.559567963466772e+02],
              [0, 2.809657898807247e+02, 1.216932104820509e+02],
              [0, 0, 1]])
D = np.array([[0.015352222035580, -0.092403852639830, 0, 0, 0.108435908257982]])
W = 320
H_ = 240
F = 10  # 焦距


class NposClass:
    def __init__(self):
        self.state = INITIAL_STATE
        self.feedbackstate = INITIAL_STATE  # 追踪状态
        self.targetx = 0  # 世界坐标系下目标点的x轴
        self.targety = 0  # 世界坐标系下目标点的y轴


class GpsClass:
    def __init__(self):
        self.state = INITIAL_STATE
        self.latitude = 0  # 纬度
        self.longitude = 0  # 经度
        self.x = 0  # 世界坐标系下x轴，东方为正方向
        self.y = 0  # 世界坐标系下y轴，北方为正方向


class ImuClass:
    def __init__(self):
        self.state = INITIAL_STATE
        self.orientation = [0, 0, 0, 0]  # 四元数
        self.angular_velocity = [0, 0, 0]  # 角速度
        self.linear_acceleration = [0, 0, 0]  # 线加速度
        self.rpy = [0, 0, 0]  # 三轴角度,单位为角度


class PwmClass:
    def __init__(self):
        self.stdbpwm = 0  # 右舷电机pwm
        self.portpwm = 0  # 左舵电机pwm


class ImgClass:
    def __init__(self):
        self.state = INITIAL_STATE
        self.img = 0
        # 当前图片时的位置
        self.rpy = 0
        self.gps = GpsClass()


class DetectClass:
    def __init__(self):
        self.state = INITIAL_STATE
        self.pred_boxes = []  # (x1, y1, x2, y2, cls_id, label_name, conf)
        self.img = 0
        self.rpy = 0
        self.gps = GpsClass()


# 读写目标位置的变量与锁
nposLock = threading.Lock()
npos = NposClass()
# GPS的变量与锁
gpsLock = threading.Lock()
gps = GpsClass()
# IMU的变量与锁
imuLock = threading.Lock()
imu = ImuClass()
# PWM的变量与锁
pwmLock = threading.Lock()
pwm = PwmClass()
# IMG的变量与锁
imgLock = threading.Lock()
img = ImgClass()
# DETECT的变量与锁
detectLock = threading.Lock()
detect = DetectClass()
