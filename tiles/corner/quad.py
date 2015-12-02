#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: quad.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import glob
import numpy as np
from datetime import datetime
import cv2
import hashlib
import json


import matplotlib.pyplot as plt
def show_img_sync(img):
    if img.shape[0] == 3 and len(img.shape) == 3 and img.shape[2] != 3:
        n = np.zeros((img.shape[1], img.shape[2], 3))
        for k in range(3):
            n[:,:,k] = img[k]
        img = n
    plt.imshow(img)
    plt.show()

class QuadDataset(object):
    def __init__(self, cnt, krange, arange):
        self.inputs = glob.glob('./quad_labeled/*.jpg')
        self.nin = len(self.inputs);
        labels = [x[:-3] + 'json' for x in self.inputs]
        self.jsons = [np.asarray(json.load(open(x))) for x in labels]

        sha = hashlib.sha256()
        sha.update(str(datetime.now()));
        self.rng = np.random.RandomState(int(sha.hexdigest()[10:20], 16) % 4294967295)
        self.krange = krange
        self.arange = arange
        self.cnt = cnt

    def get_data_stream(self):
        for id in range(self.nin):
            im = cv2.imread(self.inputs[id])
            data = self.jsons[id]
            npoints = data.shape[0] / 4
            k_s = self.rng.uniform(self.krange[0], self.krange[1], (npoints,))
            a_s = self.rng.uniform(self.arange[0], self.arange[1], (npoints,))
            for pid in xrange(npoints):
                box = data[pid * 4:pid * 4 + 4,:]
                rot = cv2.getRotationMatrix2D(
                    (np.mean(box[:,0]), np.mean(box[:,1])), a_s[pid], 1)
                pp_rot = rot.dot(np.concatenate((box, np.ones((4,1))),
                                                axis=1).transpose()).transpose()

                xmin, ymin = np.amin(pp_rot, axis=0)
                xmax, ymax = np.amax(pp_rot, axis=0)
                w = xmax - xmin
                h = ymax - ymin
                s = max([w, h]) * (1 + k_s[pid])
                x0 = xmin - (s-w) * 0.5
                y0 = ymin - (s-h) * 0.5
                trans = [-x0 + 1, -y0 + 1]
                s = int(s)

                pp_rot += trans

                rot[0,2] += trans[0]
                rot[1,2] += trans[1]
                smallim = cv2.warpAffine(im, rot, (s, s))
                yield (smallim, pp_rot) # square img and 4x2 points

if __name__ == '__main__':
    d = QuadDataset(10, [0,0.3], [0, 20])
    for k in d.get_data_stream():
        print k[1]
        show_img_sync(k[0])

