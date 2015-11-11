#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: segment.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import numpy as np
import sys
import sklearn.cluster
from copy import copy

from lib.imgproc import *

img = cv2.imread('box.png')
gray = 255 - power_togray(img)
gray = cv2.GaussianBlur(gray, (5,5), 1)

edges = cv2.Canny(gray, 20, 60)

show_img(edges)

