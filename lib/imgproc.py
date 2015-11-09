#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

def show_img_mat(img):
    if img.shape[0] == 3 and len(img.shape) == 3 and img.shape[2] != 3:
        n = np.zeros((img.shape[1], img.shape[2], 3))
        for k in range(3):
            n[:,:,k] = img[k]
        img = n
    plt.imshow(img)
    plt.show()

def show_img(img):
    cv2.imshow("", img)
    cv2.waitKey()

def power_togray(img):
    assert img.ndim == 3
    ret = np.square(255. - img.astype('float32')).sum(axis=2)
    ret = ret / ret.max() * 255
    ret = ret.astype('uint8')
    return ret

def merge_rects(rects):
    graph = defaultdict(list)
    res = []
    for r in rects:
        now_area = r.area()
        for rr in rects:
            if rr is r: continue
            if rr.area() < now_area: continue
            # r is smaller
            if r.intersect_ratio(rr) > 0.9:
                break
        else:
            res.append(r)
    return res
