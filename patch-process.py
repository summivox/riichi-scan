#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: patch-process.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys
import os
import cv2
import numpy as np

from lib.imgproc import *
from lib.rect import Rect

from boundingbox import detect_boundingbox

img = cv2.imread(sys.argv[1])
gray = 255 - power_togray(img)

gray = cv2.GaussianBlur(gray, (5,5), 1)
edges = cv2.Canny(gray, 50, 150)
show_img(edges)

comp = pca_rot(edges)
print comp
print -np.arctan2(comp[0],comp[1])

rotM = cv2.getRotationMatrix2D(
    (edges.shape[1]/2,edges.shape[0]/2),
    np.arctan2(comp[0],comp[1]) * 180 / np.pi, 1)

cv2.warpAffine(
    img, rotM,
    (edges.shape[1],edges.shape[0]),img,
    cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
show_img(img)

rects = detect_boundingbox(img)

bigR = rects[0]
for r in rects[1:]:
    bigR.update(r)
img = img[bigR.y0:bigR.y1+1,bigR.x0:bigR.x1+1]
show_img(img)

from scan import try_scan
try_scan(img)
sys.exit()


for y in range(100):
    x = y / comp[0] * comp[1]
    if x >= img.shape[1]: break
    img[y,x] = (0,0,255)

show_img(img)


