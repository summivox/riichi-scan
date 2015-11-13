#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: cut-line.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
im = cv2.imread(sys.argv[1])

#im = cv2.resize(im, (0,0), im, 1, 1.33333 / 1.7)
#cv2.imshow("", im)
#cv2.waitKey()
#sys.exit()

step = im.shape[1] / 9
width = im.shape[1] / 9

import os
import shutil
try:
    shutil.rmtree('tmp')
    os.mkdir('tmp')
except:
    pass

left = 0
right = width
while True:
    subim = im[:,left:right,:]
    name = "tmp/{:03d}.png".format(left)

    cv2.imwrite(name, subim)
    left += step
    right += step
    if right >= im.shape[1]:
        break


