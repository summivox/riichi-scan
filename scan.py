#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: scan.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import numpy as np
import os

RATIO = 1.5

def split(img, n):
    start = 0
    step = img.shape[1] / n
    ret = []

    #padx = int(0.08 * img.shape[0])
    #pady = int(0.16 * img.shape[0])
    dx = 3

    for _ in range(n):
        left = max(0, start-dx)
        right = min(img.shape[1],start+step+dx)
        subim = img[:,left:right]
        #subim = cv2.copyMakeBorder(subim, pady, pady, padx, padx, cv2.BORDER_CONSTANT, value=(0,0,0))

        ret.append(subim)
        start += step
    return ret


def try_scan(img):
    ratio = img.shape[1] * 1.0 / img.shape[0]
    n_guess = ratio * RATIO
    guesses = map(int, [np.floor(n_guess), np.ceil(n_guess)])
    print guesses

    for ntile in guesses:
        splits = split(img, ntile)
        dirname = 'split-{}'.format(ntile)
        if os.path.isdir(dirname):
            import shutil
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        for idx, k in enumerate(splits):
            fname = os.path.join(dirname, '{}.png'.format(idx))
            cv2.imwrite(fname, k)

if __name__ == '__main__':
    import sys
    img = cv2.imread(sys.argv[1])
    try_scan(img)
