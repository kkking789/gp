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
import numpy as np
import rospy
from simple_pid import PID
from sensor_msgs.msg import Imu, NavSatFix, Pose2D, Image
from queue import Queue


H = 0.02  # 左舷右舷之间的距离，单位米
INIT = -1
RUN = 1
FINISH = 2

class TrackNode:
    def __init__(self):
        self.anglepid = PID(2, 0, 1, 0, output_limits=(-1, 1))
        self.wpid = PID(2, 0, 1, 0, output_limits=(-0.4 / H, 0.4 / H))
        self.distancepid = PID(2, 0, 1, 0, output_limits=(-0.4 / 2, 0.4 / 2))

        self.posesub = rospy.Subscriber("/gpspose", Pose2D, self.posecallback, queue_size=10)
        self.imusub = rospy.Subscriber("/imu/data_raw", Imu, self.imucallback, queue_size=10)
        self.gpointsub = rospy.Subscriber("/goalpoint", Pose2D, self.gpointcallback,queue_size=2)
        self.portpub = rospy.Publisher("/gnc/port_cmd", float, queue_size=2)
        self.stdbpub = rospy.Publisher("/gnc/stdb_cmd", float, queue_size=2)

        self.rpy = []
        self.pose = []
        self.av = []
        self.gpoint = Queue(maxsize=5)
        self.targetx = 0
        self.targety = 0

        self.state = INIT

    def posecallback(self, posemsg):
        self.pose = [posemsg.x, posemsg.y, posemsg.theta]

    def imucallback(self, imumsg):
        self.av = imumsg.angular_velocity

    def gpointcallback(self, gpointmsg):
        self.gpoint.put((gpointmsg.x, gpointmsg.y))

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.pose or not self.av:
                rate.sleep()
                continue
            if self.state == INIT or self.state == FINISH:
                if len(self.gpoint) > 0:
                    goal = self.gpoint.get()
                    if goal[0] > 0 and goal[1] > 0:
                        self.targetx = goal[0]
                        self.targety = goal[1]
                        self.state = RUN
            if self.state == RUN:
                # 控制船体角速度
                self.anglepid.setpoint = math.atan2(self.targetx, self.targety)
                goalw = self.anglepid(self.pose[2])
                self.wpid.setpoint = goalw
                wpwm = self.wpid(self.av[2])
                # 控制船体线速度
                distance = math.sqrt((self.targetx - self.pose[0]) ** 2 + (self.targety - self.pose[1]) ** 2)
                vpwm = self.distancepid(distance)

                portpwm = (wpwm * H + vpwm * 2) / 2
                stdbpwm = (vpwm * 2 - wpwm * H) / 2
                portpwm = np.clip(portpwm, -1.0, 1.0)
                stdbpwm = np.clip(stdbpwm, -1.0, 1.0)

                if distance < 0.1:
                    self.state = FINISH

            else:
                portpwm = 0.2
                stdbpwm = -0.2

            self.portpub.publish(portpwm)
            self.stdbpub.publish(stdbpwm)
            rate.sleep()


if __name__ == '__main__':
    node = TrackNode()
    node.run()
