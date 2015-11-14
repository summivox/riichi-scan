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

from boundingbox import detect_boundingbox


img = cv2.imread('./photo/white-bg.png')
#img = cv2.imread('./photo/line2.png')

rects = detect_boundingbox(img)

for idx, r in enumerate(rects):
    box = img[r.y0:r.y1,r.x0:r.x1]
    cv2.imwrite("box-{}.png".format(idx), box)

