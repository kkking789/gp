"""
功能：对目标点的追踪
耦合点：
    1、当npos.feedbackstate置为FINISH时，另一个线程需要将npos.state置为RUNNING，且导入目标位置
    2、IMU与GPS的传感器数据需要完成初始化，即state被置为RUNNING时，本线程才能正常获取数据
注意事项：
    1、wgs2utm与q2e需要在接收数据的线程中完成转化
    2、目标位置是直接导入世界坐标系下的xy坐标，在处理数据的线程中需要通过视差图与视场角FOV来转化得到

时间：2025/4/2,22:45
作者：kkking789
"""
import math
import copy
from simple_pid import PID
from share import *


def tracking():
    global npos
    # 初始化变量
    tracking_state = INITIAL_STATE
    x = 0  # 船的世界坐标系x轴，东方为正方向
    y = 0  # 船的世界坐标系y轴，北方为正方
    yaw = 0  # 船的世界坐标系下yaw轴，单位角度
    # 初始化pid
    anglepid = PID(2, 0, 1, 0, output_limits=(-1, 1))
    wpid = PID(2, 0, 1, 0, output_limits=(-0.4 / H, 0.4 / H))
    distancepid = PID(2, 0, 1, 0, output_limits=(-0.4 / 2, 0.4 / 2))
    while True:
        # 各类传感器数据获取
        with nposLock:
            # npos状态置RUNNIG，目标位置装填---->获取目标位置，npos状态置STOP
            # ----->完成追踪，npos的feedback置FINISH---->npos状态置RUNING，填装新的目标位置
            if npos.state == RUNNING_STATE:
                targetx = copy.copy(npos.targetx)
                targety = copy.copy(npos.targety)
                npos.state = STOP_STATE
                tracking_state = RUNNING_STATE
            npos.feedbackstate = tracking_state
        with gpsLock:
            if gps.state == RUNNING_STATE:
                x = copy.copy(gps.x)
                y = copy.copy(gps.y)
        with imuLock:
            if imu.state == RUNNING_STATE:
                av = copy.copy(imu.angular_velocity)
                yaw = copy.copy(imu.rpy[2])

        # 开始追踪点
        if tracking_state == RUNNING_STATE:
            # 控制船体角速度
            anglepid.setpoint = math.atan2(targetx, targety)
            goalw = anglepid(yaw)
            wpid.setpoint = goalw
            wpwm = wpid(av[2])
            # 控制船体线速度
            distance = math.sqrt((  - x) ** 2 + (targety - y) ** 2)
            vpwm = distancepid(distance)

            portpwm = (wpwm * H + vpwm * 2) / 2
            stdbpwm = (vpwm * 2 - wpwm * H) / 2
            with pwmLock:
                pwm.portpwm = portpwm
                pwm.stdbpwm = stdbpwm

            if distance < 0.1:
                tracking_state = FINISH_STATE
        elif tracking_state == FINISH_STATE or tracking_state == INITIAL_STATE:
            # 完成追踪并无新追踪点时自旋，直到找到新点
            with pwmLock:
                pwm.portpwm = 0.2
                pwm.stdbpwm = -0.2


class TrackThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        tracking()
