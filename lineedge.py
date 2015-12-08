#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: lineedge.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import sys
import cv2

from lib.imgproc import *
from nnlib import driver

driver.load_model_fcn(
    os.path.join(os.path.dirname(__file__), 'nnlib', 'fcn.pkl'))

def line_detect(img):
    RUN_SIZE = 40
    h = img.shape[0]
    ratio = float(h) / RUN_SIZE
    neww = int(float(img.shape[1]) / ratio)
    smallim = cv2.resize(img, (neww, RUN_SIZE))

    mask = np.zeros(smallim.shape[:2])

    inputs = []
    for start in range(neww - RUN_SIZE):
        crop = smallim[:,start:start+RUN_SIZE]
        inputs.append(crop)
    heatmaps = driver.detect(inputs)
    for start in range(neww - RUN_SIZE):
        mask[:,start:start+RUN_SIZE] += heatmaps[start]
    show_img_mat(mask)


if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])
    line_detect(im)
