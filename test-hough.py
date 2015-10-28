#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import cv2
import numpy as np

from lib.imgproc import show_img


img = cv2.imread('./photo/line2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)
show_img(img)

