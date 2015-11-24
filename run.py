#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import numpy as np

import argparse
import logging
import os
import shutil
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from lib.imgproc import *
from scan import try_scan

def set_logger(args):
    try:
        os.mkdir(args.logdir)
    except OSError:
        pass
    hdl = logging.FileHandler(
        filename=os.path.join(args.logdir, 'log.log'),
        mode='w', encoding='utf-8')
    logger.addHandler(hdl)
    logger.addHandler(logging.StreamHandler())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image file')
    parser.add_argument('-l', '--logdir', default='logs',
                        help='directory to write log')
    ret = parser.parse_args()
    set_logger(ret)
    return ret

def log_img(name, img):
    cv2.imwrite(os.path.join(args.logdir, name + '.png'), img)

def thresh_segmented_img(img):
    assert img.ndim == 3
    mingray = np.min(img, axis=2)
    _, mask = cv2.threshold(mingray, 200, 255, cv2.THRESH_BINARY)
    return mask


if __name__ == '__main__':
    global args
    args = get_args()

    img = cv2.imread(args.input)

# segmentation:
    #from vendor.waveseg import segment
    #seg_img = segment(img)
    #log_img('segmented', seg_img)
    seg_img = cv2.imread(os.path.join(args.logdir, 'segmented.png'))

# threshold
    mask = thresh_segmented_img(seg_img)
    log_img('segmask', mask)

# shrink and rotate:
    bbox = shrink_binary_img(mask)
    mask = bbox.roi(mask)
    img = bbox.roi(img)
    bbox_shape = (mask.shape[1], mask.shape[0])

    rotMatrix, comp = pca_getM(mask)
    logger.info("PCA: " + str(comp))
    cv2.warpAffine(mask, rotMatrix, bbox_shape, mask,
                  cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
    cv2.warpAffine(img, rotMatrix, bbox_shape, img,
                  cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
    log_img('rotated1', img)

# estimate height
    heights = np.asarray(all_height(mask), dtype='int')
    median_height = np.median(heights)
    logger.info("Median height: " + str(median_height))


