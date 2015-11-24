#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys

import argparse

from vendor.waveseg import segment


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image file')
    parser.add_argument('-l', '--logdir', help='directory to write log')
    return parser.parse_args()




if __name__ == '__main__':
    global args
    args = get_args()

    img = cv2.imread(args.input)

#1: segmentation
    seg_img = segment(img)
    cv2.imwrite(os.path.join(args.logdir, 'segmented.png'), seg_img)


