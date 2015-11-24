#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from collections import defaultdict
import copy
from rect import Rect

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

def power_togray(img):
    assert img.ndim == 3
    ret = np.square(255. - img.astype('float32')).sum(axis=2)
    ret = ret / ret.max() * 255
    ret = ret.astype('uint8')
    return ret

def merge_rects(rects):
    graph = defaultdict(list)
    res = []
    for r in rects:
        now_area = r.area()
        for rr in rects:
            if rr is r: continue
            if rr.area() < now_area: continue
            # r is smaller
            if r.intersect_ratio(rr) > 0.9:
                break
        else:
            res.append(r)
    return res

def draw_rects(img, rects):
    ret = copy.copy(img)
    if ret.ndim == 2:
        ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
    for r in rects:
        pt1 = (r.x0, r.y0)
        pt2 = (r.x1, r.y1)
        cv2.rectangle(ret, pt1, pt2, (0,0,255), 1)
    return ret

def pca_rot(img):
    idxs = np.nonzero(img)
    idxs = np.transpose(idxs)
    pca = PCA(n_components=1)
    pca.fit(idxs)
    # dy, dx
    return pca.components_[0]

def pca_getM(img):
    comp = pca_rot(img)
    if comp[1] < 0:
        comp = comp * -1
    rotM = cv2.getRotationMatrix2D(
        (img.shape[1]/2,img.shape[0]/2),
        np.arctan2(comp[0],comp[1]) * 180 / np.pi, 1)
    return rotM, comp

def shrink_binary_img(img):
    # return a rect
    idxs = np.nonzero(img)
    ymin, ymax = idxs[0].min(), idxs[0].max()
    xmin, xmax = idxs[1].min(), idxs[1].max()
    return Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1)

def all_height(img):
    #img: binary img
    w = img.shape[1]
    ret = []
    for x in range(w):
        col = img[:,x]
        nz = np.nonzero(col)[0]
        if len(nz) < 2:
            ret.append(0)
        else:
            nz = nz[-1] - nz[0]
            ret.append(nz)
            #print nz
    return ret

def snap_up(img):
    w = img.shape[1]
    ret = np.zeros(img.shape)
    for x in range(w):
        col = img[:,x]
        nz = np.nonzero(col)[0]
        if len(nz) < 2:
            continue
        else:
            nnz = nz[-1] - nz[0]
            ret[:nnz,x] = img[nz[0]:nz[-1],x]
    return ret
