import cv2
import time
import os
import sys
import importlib
import time
import shutil
import argparse
import yaml
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
sys.path.append(BASE_PATH)
importlib.reload(sys)


def inference_single_mat(det, thresh, src_mat, cls, img_size=(640, 640), draw_mat=None):
    # 实例化V5检测类，可指定检测阈值，输入图片，输出图片，需要检测类别，画框颜色，以及使用的gpuid
    t1 = time.time()
    img = src_mat
    # 模型推理
    img_res, det_res = det.detect(img, cls, thresh, img_size=img_size)
    t2 = (time.time() - t1) * 1000
    print("inference time:{} ms".format(t2))
    # 绘制模型检测到的框
    img_res = det.draw_box(img, det_res)
    if draw_mat is not None:
        cols, rows, _ = img.shape
        dcols, drows, _ = draw_mat.shape
        draw_mat[dcols-cols:dcols, drows-rows:drows] = img
    # 打印模型检测到的框信息
    det.print_result(det_res)
    return img_res, det_res


def get_cls(yaml_path):
    # 获取yaml中的检测类别
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    cls = data.get('names', [])
    return cls


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/best.pt')
    parser.add_argument('--todetect', type=str, default='./todetect')
    parser.add_argument('--detected', type=str, default='./detected')
    parser.add_argument('--yaml', type=str, default='./yaml/data.yaml')

    return parser.parse_known_args()[0]


class yolo5_arr:
    def __init__(self, det, tresh=0.3, yaml='./yaml/data.yaml'):
        self.det = det
        self.tresh = tresh
        self.yaml = yaml

    def detecting(self, src_mat, img_size=(640, 640), draw_mat=None):
        cls = get_cls(self.yaml)
        return inference_single_mat(self.det, self.tresh, src_mat, cls, img_size=img_size, draw_mat=draw_mat)
