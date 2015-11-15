#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-neg.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys
import cv2
import os

linef = sys.argv[1]
n = int(sys.argv[2])

img = cv2.imread(linef)
start = img.shape[1] / n * 0.5
step = img.shape[1] / n
while True:
    right = int(start + step + 1)
    if right >= img.shape[1]:
        break
    subim = img[:,start:right]
    fname = 'neg-g{}.png'.format(right)
    if os.path.exists(fname):
        fname = str(id(fname)) + fname
    cv2.imwrite(fname, subim)
    start += step

