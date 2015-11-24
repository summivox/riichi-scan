#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-seg.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import numpy as np
from lib.imgproc import *
import sys

orig_img = cv2.imread('photo/aligned1-crop.png')
img = cv2.imread('photo/aligned1-ws-crop.png')

mingray = np.min(img, axis=2)
_, gray = cv2.threshold(mingray, 200, 255, cv2.THRESH_BINARY)

# TODO filter out white region on the border

bb = shrink_binary_img(gray)
gray = gray[bb.y0:bb.y1+1,bb.x0:bb.x1+1]
orig_img = orig_img[bb.y0:bb.y1+1,bb.x0:bb.x1+1]
show_img(gray)

rotM = pca_getM(gray)

cv2.warpAffine(
    gray, rotM,
    (gray.shape[1],gray.shape[0]),gray,
    cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
cv2.warpAffine(
    orig_img, rotM,
    (gray.shape[1],gray.shape[0]),orig_img,
    cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
show_img(gray)

heights = np.asarray(all_height(gray), dtype='int')
median_height = np.median(heights)
print median_height

height_diff_thres = median_height * 0.2

w = gray.shape[1]
# TODO: use height to re-partition
for k in range(w):
    if abs(heights[k] - median_height) > height_diff_thres:
        gray[:,k] = 0

bb = shrink_binary_img(gray)
pad = median_height / 1.4 * 0.1
left = max(0, bb.x0 - pad)
right = min(bb.x1 + 1 + pad, orig_img.shape[1])
orig_img = orig_img[bb.y0:bb.y1+1,left:right]
gray = gray[bb.y0:bb.y1+1,left:right]
show_img(orig_img)

rotM = pca_getM(gray)
cv2.warpAffine(
    gray, rotM,
    (gray.shape[1],gray.shape[0]),gray,
    cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
cv2.warpAffine(
    orig_img, rotM,
    (gray.shape[1],gray.shape[0]),orig_img,
    cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)


show_img(orig_img)
cv2.imwrite("box.png", orig_img)
