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


#img = cv2.imread('./photo/white-bg.png')
img = cv2.imread('./photo/line2.png')
# TODO judge background color, if white, use less blur, important
gray = 255 - power_togray(img)
gray = cv2.GaussianBlur(gray, (5,5), 1)

#show_img(gray)
edges = cv2.Canny(gray, 50, 150)

#show_img(edges)

dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dilated = cv2.dilate(edges, dil_kern)
show_img(dilated)

contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0]

boxes = [cv2.boundingRect(k) for k in contours]
rects = [Rect(*b) for b in boxes]

rects = merge_rects(rects)
rects = filter_noise(rects)

empty = np.zeros(edges.shape, dtype='uint8')
empty = img
for r in rects:
    pt1 = (r.x0, r.y0)
    pt2 = (r.x1, r.y1)
    cv2.rectangle(empty, pt1, pt2, (255,0,0), 1)

show_img(empty)

#from IPython import embed; embed()

#mser = cv2.MSER(_max_area=15000, _max_variation=0.25)
#regions = mser.detect(gray)
