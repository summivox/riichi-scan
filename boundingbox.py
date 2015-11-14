#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: boundingbox.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
from lib.imgproc import *
from lib.rect import Rect

def detect_boundingbox(img):

## REQUIRE an input like a line,
## and tiles lined-up with similar height

    def filter_noise(rects):
        rects = [r for r in rects if \
            r.h > 50 and r.w > 50       # minimum allowed size for a box
                ]
        return rects

# TODO judge background color, if white, use less blur, no power, important
    gray = (255 - power_togray(img)).astype('float32')
    gray = np.power(gray / 255.0, 1.5)
    gray = cv2.GaussianBlur(gray, (5,5), 1)
    #gray = cv2.GaussianBlur(gray, (5,5), 0.8)

    show_img(gray)
    gray = (gray * 255).astype('uint8')
    edges = cv2.Canny(gray, 50, 150)

    show_img(edges)
    #import sys
    #sys.exit()

    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(edges, dil_kern)
    show_img(dilated)

    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]

    boxes = [cv2.boundingRect(k) for k in contours]
    rects = [Rect(*b) for b in boxes]

    rects = merge_rects(rects)
    rects = filter_noise(rects)

# TODO expand rect

#show_img(draw_rects(img, rects))

    return rects

if __name__ == '__main__':
    import cv2
    import sys

    img = cv2.imread(sys.argv[1])
#img = cv2.imread('./photo/line2.png')

    rects = detect_boundingbox(img)

    vis = draw_rects(img, rects)
    show_img(vis)

    for idx, r in enumerate(rects):
        box = img[r.y0:r.y1,r.x0:r.x1]
        cv2.imwrite("box-{}.png".format(idx), box)

