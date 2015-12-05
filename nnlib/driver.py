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

def mean_subtract(img):
    img = img.astype('float32')
    mean = img.mean(axis=0).mean(axis=0)
    return img - mean

def load_model(filename):
    global model
    model = pickle.load(open(filename))
    model.finish()
    return model

def predict(img):
    assert img.shape[:2] == (70, 50)
    img = mean_subtract(img)
    img = img * 2.0 / 255 - 1
    prob = model.run_only_last(img)[0]
    pred = np.argmax(prob)
    return prob, pred

if __name__ == '__main__':
    load_model('model.mdl')

    import glob
    #filelist = glob.glob('./alt1/*.png')
    filelist = ['/tmp/1204-18:17:02.png']
    TILES = ['1m','1p','1s','1z','2m','2p','2s','2z','3m','3p','3s','3z','4m','4p','4s','4z','5m','5p','5s','5z','6m','6p','6s','6z','7m','7p','7s','7z','8m','8p','8s','9m','9p','9s', 'neg']
    for f in filelist:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (50, 70))
        prob, pred = predict(img)
        print os.path.basename(f), pred, TILES[pred]

