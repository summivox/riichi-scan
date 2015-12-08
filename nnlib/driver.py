#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: driver.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cPickle as pickle
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import network_runner
from imageutil import get_image_matrix

import matplotlib.pyplot as plt
def show_img_sync(img):
    if img.shape[0] == 3 and len(img.shape) == 3 and img.shape[2] != 3:
        n = np.zeros((img.shape[1], img.shape[2], 3))
        for k in range(3):
            n[:,:,k] = img[k]
        img = n
    plt.imshow(img)
    plt.show()

class BatchPredictor(object):
    BATCHES = [1, 8, 16, 64]
    def __init__(self, pkl_filename):
        self.fname = pkl_filename
        self.init_nns()

    def init_nns(self):
        self.nns = {}
        for k in self.BATCHES:
            self.nns[k] = network_runner.get_nn(self.fname, k)
        self.input_shape = self.nns[1].nn.input_shape[2:]

    def run_batch(self, imgs):
        nr_img = len(imgs)
        assert all([img.shape[:2] == self.input_shape for img in imgs])
        def preprocess(img):
            img = mean_subtract(img)
            return img * 2.0 / 255 - 1
        # to ic01
        images_to_run = np.asarray(
            [get_image_matrix(preprocess(m), False) for m in imgs])

        results = []
        nowid = 0
        for k in reversed(self.BATCHES):
            nn = self.nns[k]
            while nr_img >= k:
                print k
                inputs = images_to_run[nowid:nowid+k]
                inputs = inputs.reshape((k, -1))
                outputs = nn.func(inputs)
                results.append(outputs)
                nr_img -= k
                nowid += k
        results = np.concatenate(results, axis=0)
        return results


def mean_subtract(img):
    img = img.astype('float32')
    mean = img.mean(axis=0).mean(axis=0)
    return img - mean

def load_model_recog(filename):
    global model_recog
    model_recog = BatchPredictor(filename)
def predict(imgs):
    prob = model_recog.run_batch(imgs)  # B x N_TILE
    pred = np.argmax(prob, axis=1)  # Bx1
    return prob, pred

def load_model_fcn(filename):
    global model_fcn
    model_fcn = BatchPredictor('fcn.pkl')

def detect(imgs):
    prob = model_fcn.run_batch(imgs)
    prob = prob[:,0,:,:]
    return prob

if __name__ == '__main__':
    import glob
    #load_model_recog('recog.pkl')
    #filelist = glob.glob('./alt1/*.png')
    #TILES = ['1m','1p','1s','1z','2m','2p','2s','2z','3m','3p','3s','3z','4m','4p','4s','4z','5m','5p','5s','5z','6m','6p','6s','6z','7m','7p','7s','7z','8m','8p','8s','9m','9p','9s', 'neg']
    #imgs = []
    #for f in filelist:
        #img = cv2.imread(f, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (50, 70))
        #imgs.append(img)
    #prob, pred = predict(imgs)
    #for idx, f in enumerate(filelist):
        #print os.path.basename(f), pred[idx], TILES[pred[idx]]

    load_model_fcn('fcn.mdl')
    filelist = glob.glob('../edge-test/*')
    imgs = []
    for f in filelist:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (40, 40))
        imgs.append(img)
    output = detect(imgs)
    print output.shape

    for img, o in zip(imgs, output):
        show_img_sync(img)
        show_img_sync(o)

