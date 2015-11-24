#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: waveseg.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import os

import matlab
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(
    os.path.join(os.path.dirname(__file__),
                 'include/waveseg'))

def segment(img):
    mm = matlab.uint8(img.tolist())
    res = eng.waveseg_color_prl07(mm)
    res = np.asarray(res, dtype='uint8')
    return res

if __name__ == '__main__':
    import sys
    import cv2
    I = cv2.imread(sys.argv[1])
    res = segment(I)
    cv2.imshow("", res)
    cv2.waitKey(0)

