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
import operator
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from lib.imgproc import *
import scan

import nnlib.driver as nn_driver

TILES = ['1m','1p','1s','1z','2m','2p','2s','2z','3m','3p','3s','3z','4m','4p','4s','4z','5m','5p','5s','5z','6m','6p','6s','6z','7m','7p','7s','7z','8m','8p','8s','9m','9p','9s', 'neg']
nn_driver.load_model(os.path.join(os.path.dirname(__file__), 'nnlib',
                                     'model.mdl'))
TILE_RATIO_RANGE = (1.4, 1.7)

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

def get_segmented_img():
    # XXX use cached one
    segf = args.input[:-4] + '_seg.png'
    if os.path.isfile(segf):
        seg_img = cv2.imread(segf)
        log_img('segmented', seg_img)
    else:
        from vendor.waveseg import segment
        seg_img = segment(img)
        log_img('segmented', seg_img)
        cv2.imwrite(segf, seg_img)
    return seg_img

def thresh_segmented_img(img):
    assert img.ndim == 3
    mingray = np.min(img, axis=2)
    _, mask = cv2.threshold(mingray, 170, 255, cv2.THRESH_BINARY)
    return mask

def noise_removal(mask):
    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(mask, dil_kern)
    log_img('segmask-dilate', dilated)

    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = [cv2.boundingRect(k) for k in contours[0]]
    rects = [Rect(b[0], b[1], b[2] + 1, b[3] + 1) for b in boxes]
    maxh = max([r.h for r in rects])
    valid_rects = [r for r in rects if \
        r.h > maxh * 0.6 \
       and r.w > img.shape[1] * 0.3] # minimum allowed size for a box
    logger.info("Valid bounding boxes: " + str(valid_rects))

    valid_mask = np.zeros(mask.shape, mask.dtype)
    for r in valid_rects:
        roi = r.roi(valid_mask)
        roi.fill(1)
    mask = mask * valid_mask
    log_img('segmask-filtered', mask)
    return mask

def rotate(mask, img):
    bbox_shape = (mask.shape[1], mask.shape[0])
    rotMatrix, comp = pca_getM(mask)
    logger.info("PCA: " + str(comp))
    cv2.warpAffine(mask, rotMatrix, bbox_shape, mask,
                  cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
    cv2.warpAffine(img, rotMatrix, bbox_shape, img,
                  cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
    return (mask, img)

def recog_batch(imgs):
    ret = []
    for img in imgs:
        #pady = int(0.08 * img.shape[0])
        #img = cv2.copyMakeBorder(img, pady, pady, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        #show_img_mat(img)

        img = cv2.resize(img, (50, 70))
        prob, pred = nn_driver.predict(img)
        ret.append((TILES[pred], prob[pred]))
    return ret

def select_guess(results):
# count neg
    neg_cnt = [[x[0] for x in r].count('neg') for r in results]
    logger.info("Negative class cnt: " + str(neg_cnt))
    min_neg = min(neg_cnt)
    candidates = [idx for idx, neg in enumerate(neg_cnt) if neg == min_neg]
    results = [results[k] for k in candidates]
    mean_scores = [np.mean([x[1] for x in r]) for r in results]
    logger.info("Recognition mean scores: " + str(mean_scores))
    max_score_idx = max(enumerate(mean_scores), key=operator.itemgetter(1))[0]
    return results[max_score_idx]

if __name__ == '__main__':
    global args
    args = get_args()

    img = cv2.imread(args.input)

# segmentation:
    seg_img = get_segmented_img()

# threshold
    mask = thresh_segmented_img(seg_img)
    log_img('segmask', mask)

    mask = noise_removal(mask)
    if np.sum(mask) == 0:
        logger.error("Nothing found.")
        sys.exit()
    # TODO perspective correction

# shrink and rotate:
    bbox = shrink_binary_img(mask)
    mask = bbox.roi(mask)
    img = bbox.roi(img)

    [mask, img] = rotate(mask, img)
    log_img('rotated1', img)

# estimate height
    heights = np.asarray(all_height(mask), dtype='int')
    median_height = np.median(heights)
    logger.info("Median height: " + str(median_height))

# pass2: filter by height
    height_diff_thres = median_height * 0.2
    w = mask.shape[1]
    for k in range(w):
        if abs(heights[k] - median_height) > height_diff_thres:
            mask[:,k] = 0
    bbox = shrink_binary_img(mask)

    pad = median_height / 1.4 * 0.1
    bbox.x = max(0, bbox.x0 - pad)
    bbox.w = min(bbox.x1 + 1 + pad, w) - bbox.x + 1
# TODO expand bbox up a little

    mask = bbox.roi(mask)
    img = bbox.roi(img)
    log_img('mask2', mask)

    [mask, img] = rotate(mask, img)
    log_img('rotated2', img)

    large_ratio = img.shape[1] * 1.0 / img.shape[0]

    n_guess_min = int(large_ratio * TILE_RATIO_RANGE[0])
    n_guess_max = int(large_ratio * TILE_RATIO_RANGE[1])
    logger.info("Guess number: {}~{}".format(n_guess_min, n_guess_max))
    ress = []
    for n_tile in range(n_guess_min, n_guess_max+1):
        splits = scan.split(img, n_tile)
        res = recog_batch(splits)
        ress.append(res)
        logger.info("Ans for n_tile={}: {}".format(n_tile, str(res)))
    res = select_guess(ress)
    print res
