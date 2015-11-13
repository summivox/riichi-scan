#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: scan-line.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
im = cv2.imread(sys.argv[1])

im = cv2.resize(im, (0,0), im, 1, 1.33333 / 1.7)
#cv2.imshow("", im)
#cv2.waitKey()
#sys.exit()

step = 2    #pixel
width = im.shape[0] / 1.1

import os
import shutil
try:
    shutil.rmtree('tmp')
    os.mkdir('tmp')
except:
    pass

padx = int(0.08 * im.shape[0])
pady = int(0.16 * im.shape[0])

left = 0
right = width
while True:
    subim = im[:,left:right,:]
    name = "tmp/{:03d}.png".format(left)

    subim = cv2.copyMakeBorder(subim, pady, pady, padx, padx, cv2.BORDER_CONSTANT, value=(0,0,0))

    cv2.imwrite(name, subim)
    left += step
    right += step
    if right >= im.shape[1]:
        break


