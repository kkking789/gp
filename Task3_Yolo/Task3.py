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
import Task3_Yolo.yolo5_detect as yolo5_detect
import cv2
import time

from share import *
from YoloV5Detector.V5Detector import Detector

weights = './Task3_Yolo/weights/best.pt'
yaml = './Task3_Yolo/yaml/data.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detecting():
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
    print(f"weights:{weights},yaml:{yaml},device:{device}")
    global detect
    # 网络初始化
    det = Detector(weights, yaml, device=device)
    yolo5_module = yolo5_detect.yolo5_arr(det, yaml=yaml)
    pre_time = time.time()

    while True:
        fps = 1 / (time.time() - pre_time)  # 计算实时帧数
        with imgLock:
            if img.state == RUNNING_STATE:
                todetect = copy.deepcopy(img.img)
                rpy = copy.copy(img.rpy)
                gps_ = copy.copy(img.gps)
            pre_time = time.time()
        # 数据预处理
        todetect = cv2.cvtColor(todetect, cv2.COLOR_BGR2RGB)
        todetect = cv2.convertScaleAbs(todetect, alpha=1.2, beta=0)
        # 推理
        dst_mat, det_res = yolo5_module.detecting(todetect, img_size=(640, 640))
        cv2.putText(dst_mat, f"fps:{fps}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)
        cv2.imshow("dst", dst_mat)
        cv2.waitKey(1)

        # 推理结果导出
        with detectLock:
            detect.state = RUNNING_STATE
            detect.pred_boxes = det_res
            detect.img = img
            detect.yaw = rpy
            detect.gps = gps_


class DetectThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        detecting()
