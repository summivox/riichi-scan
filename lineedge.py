#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: lineedge.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import sys
import cv2
import copy

from skimage import measure

from lib.imgproc import *
from nnlib import driver

driver.load_model_fcn(
    os.path.join(os.path.dirname(__file__), 'nnlib', 'fcn.pkl'))

def get_edge_mask(img):
    RUN_SIZE = 40
    SCAN_STEP = 2
    h = img.shape[0]
    ratio = float(h) / RUN_SIZE
    neww = int(float(img.shape[1]) / ratio)
    smallim = cv2.resize(img, (neww, RUN_SIZE))


    padx = 20
    padded = cv2.copyMakeBorder(smallim, 0,0,padx,padx,
                                cv2.BORDER_REFLECT,value=(0,0,0))
    neww += 2 * padx
    mask = np.zeros(padded.shape[:2])

    inputs = []
    for start in range(0, neww - RUN_SIZE, SCAN_STEP):
        crop = padded[:,start:start+RUN_SIZE]
        inputs.append(crop)
    heatmaps = driver.detect(inputs)
    for start in range(0, neww - RUN_SIZE, SCAN_STEP):
        mask[:,start:start+RUN_SIZE] += heatmaps[start/SCAN_STEP]
    mask = mask / mask.max() * 255
    mask = mask.astype('uint8')
    mask = mask[:,padx:-padx]
    #show_img_mat(mask)
    mask = cv2.resize(mask, (img.shape[1],img.shape[0]))
    cv2.imwrite('edgemask.png', mask)
    return mask

def process_edge_mask(img, mask):
    assert img.shape[:2] == mask.shape, "{}!={}".format(str(img.shape),
                                                        str(mask.shape))
    ssize = int(float(img.shape[0]) / 40 * 3)

    struc1 = cv2.getStructuringElement(cv2.MORPH_RECT, (ssize,1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc1)
    struc2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,ssize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc2)

    _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
# transformed mask
    #show_img_mat(mask)
    return mask

def get_blocks(mask):
    L = measure.label(mask)
    nL = L.max()

    #colorized = np.zeros(mask.shape + (3,), dtype='uint8')
    #for k in range(nL):
        #colorized[L==k] = (np.random.rand()*255,np.random.rand()*255,np.random.rand() * 255)
    #show_img_mat(colorized)

    ccs = []
    for k in range(nL):
        pts = np.nonzero(L==k)
        pts = np.asarray(pts).transpose().astype('float32')
        ccs.append(pts.reshape(pts.shape[0],1,2))
    rects = [(cv2.boundingRect(cc), cc) for cc in ccs]
    rects = [(Rect(b[1], b[0], b[3] + 1, b[2] + 1), a) for (b, a) in rects]
    #from IPython import embed; embed()
    maxw = max([r[0].w for r in rects])
    rects = [r for r in rects if r[0].w < maxw * 0.8]
    maxh = max([r[0].h for r in rects])
    valid_rects = [r for r in rects if r[0].h > maxh * 0.6 and
                  r[0].w > maxh * 0.3]
    for r in valid_rects:
        m = copy.copy(mask)
        ret = draw_rects(m, [r[0]])
        show_img_mat(ret)

if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])

    # step 1
    #get_edge_mask(im)

    # step 2
    mask = cv2.imread('edgemask.png', cv2.IMREAD_GRAYSCALE)
    mask = process_edge_mask(im, mask)

    get_blocks(mask)

