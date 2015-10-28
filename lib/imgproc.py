#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import matplotlib.pyplot as plt
import cv2
def show_img_mat(img):
    if img.shape[0] == 3 and len(img.shape) == 3 and img.shape[2] != 3:
        n = np.zeros((img.shape[1], img.shape[2], 3))
        for k in range(3):
            n[:,:,k] = img[k]
        img = n
    plt.imshow(img)
    plt.show()

def show_img(img):
    cv2.imshow("", img)
    cv2.waitKey()
