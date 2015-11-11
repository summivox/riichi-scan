#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-binary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import numpy as np
import sys
import sklearn.cluster
from copy import copy

from lib.imgproc import *
from lib.rect import Rect

## REQUIRE an input like a line,
## and tiles lined-up with similar height


def filter_noise(rects):
    rects = [r for r in rects if \
        r.h > 50 and r.w > 50       # minimum allowed size for a box
            ]
    return rects


img = cv2.imread('./photo/white-bg.png')
img = cv2.imread('./photo/line2.png')
# TODO judge background color, if white, use less blur, important
gray = 255 - power_togray(img)
gray = cv2.GaussianBlur(gray, (5,5), 1)

#show_img(gray)
edges = cv2.Canny(gray, 50, 150)

#show_img(edges)

dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dilated = cv2.dilate(edges, dil_kern)
#show_img(dilated)

contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0]

boxes = [cv2.boundingRect(k) for k in contours]
rects = [Rect(*b) for b in boxes]

rects = merge_rects(rects)
rects = filter_noise(rects)

#show_img(draw_rects(img, rects))

r = rects[0]
box = img[r.y0:r.y1,r.x0:r.x1]
cv2.imwrite("box.png", box)

