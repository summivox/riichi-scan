#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run2.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import numpy as np
import copy

import argparse
import logging
import os
import shutil
import operator
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from sklearn.decomposition import PCA

from lib.imgproc import *
import scan

import nnlib.driver as nn_driver
import lineedge

TILES = ['1m','1p','1s','1z','2m','2p','2s','2z','3m','3p','3s','3z','4m','4p','4s','4z','5m','5p','5s','5z','6m','6p','6s','6z','7m','7p','7s','7z','8m','8p','8s','9m','9p','9s', 'neg']
nn_driver.load_model_recog(os.path.join(os.path.dirname(__file__), 'nnlib',
                                     'recog.pkl'))
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
        logger.info("Running matlab... this is slow")
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
    inputs = []
    for img in imgs:
        #pady = int(0.08 * img.shape[0])
        #img = cv2.copyMakeBorder(img, pady, pady, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        #show_img_mat(img)

        img = cv2.resize(img, (50, 70))
        inputs.append(img)
    probs, preds = nn_driver.predict(inputs)
    for pred, prob in zip(preds, probs):
        ret.append((TILES[pred], prob[pred]))
    return ret

def get_roi_from_edgeblock(blocks, line):
    rois = []
    for r, pts in blocks:
        pts = pts[:,0,:]
        r.expand(0.05, 0.15)
        roi = r.safe_roi(line)
        rois.append(roi)
        #show_img_mat(roi)

        #hull = cv2.convexHull(pts).astype('int32')
        #empty = np.zeros(mask.shape)
        #cv2.fillConvexPoly(empty, hull, 1)
        #pts = np.asarray(empty.nonzero()).transpose()[:,::-1]
        #show_img_mat(empty)
        #pca = PCA(n_components=1)
        #pca.fit(pts)
        #comp = pca.components_[0]
        #print comp
        #rotM, comp = pca_getM_from_comp(r.center(), comp)
        #rotM = cv2.invertAffineTransform(rotM)
        #pts3 = np.concatenate((pts, np.ones((pts.shape[0],1))), axis=1)
        #transformed = np.dot(rotM, pts3.transpose()).transpose()    # nx2
        #transformed = transformed.reshape((transformed.shape[0],1,2))
        #newbox = cv2.boundingRect(transformed.astype('float32'))
        #newbox = (max(x,0) for x in newbox)
        #newrect = Rect(newbox[0], newbox[1], newbox[2] + 1, newbox[3] + 1)
        #newrect = r
        #newline = copy.copy(expand_line)
        #cv2.warpAffine(expand_line, rotM, (expand_line.shape[1],expand_line.shape[0]),
                       #newline, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        #m = draw_rects(newline, [newrect])
    return rois

def tile_shape_analysis(rois):
    shapes = [roi.shape[:2] for roi in rois]
    median = np.median(shapes, axis=0)
    print median
    print shapes

    newrois = []
    for r in rois:
        if r.shape[1] > median[1] * 1.8:    # two tiles connected
            mid = r.shape[1] / 2
            pad = mid * 0.1
            # split
            newrois.append(r[:,:mid+pad])
            newrois.append(r[:,mid-pad:])
            continue
        newrois.append(r)

    # rotate
    def maybe_rotate(img):
        if img.shape[0] > img.shape[1]:
            return img
        return np.rot90(img)
    newrois = [maybe_rotate(r) for r in newrois]
    return newrois

if __name__ == '__main__':
    global args
    args = get_args()

    img = cv2.imread(args.input)
    orig_img = img

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

    [mask, img] = rotate(mask, img)
    bbox = shrink_binary_img(mask)
    log_img('rotated1', img)

    # to test line edge
    bbox.expand(0.05, 0.12)
    expand_line = bbox.safe_roi(img)
    log_img('expand_line', expand_line)

    #edge_mask = lineedge.get_edge_mask(expand_line)
    edge_mask = cv2.imread('edgemask.png', cv2.IMREAD_GRAYSCALE)
    assert edge_mask.shape == expand_line.shape[:2], edge_mask.shape
    log_img('edge_mask_raw', edge_mask)

    mask = lineedge.process_edge_mask(expand_line, edge_mask)
    log_img('morph_edge_mask', mask)
    blocks = lineedge.get_blocks(mask)   # rect, points
    rois = get_roi_from_edgeblock(blocks, expand_line)

    rois = tile_shape_analysis(rois)

    for idx,r in enumerate(rois):
        log_img('tile{}'.format(idx), r)

    results = recog_batch(rois)
    print results


