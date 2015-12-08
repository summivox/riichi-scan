#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave, toimage, imread
import theano.tensor as T
import theano
import sys, gzip
import cPickle as pickle
import operator
import itertools
from itertools import count, izip
import time

from network_trainer import NNTrainer
from imageutil import tile_raster_images, get_image_matrix
from layers.layers import *

class NetworkRunner(object):
    def __init__(self, input_shape):
        """ input size in (height, width)"""
        # nn is the underlying neural network object to run with
        self.nn = NNTrainer(input_shape)

    def finish(self, only_last=True):
        """ compile the output of each layer as theano function"""
        print "Compiling..."
        last = self.nn.layers[-1]
        if last.NAME == 'conv':
            # HACK for fcn
            self.func = theano.function([self.nn.x],
                                 T.nnet.sigmoid(last.output_test),
                                allow_input_downcast=True)
        else:
            self.func = theano.function([self.nn.x],
                                 last.p_y_given_x,
                                allow_input_downcast=True)

    def _prepare_img_to_run(self, img):
        assert self.nn.batch_size == 1, \
                "batch_size of runner is not 1, but trying to run against 1 image"
        img = get_image_matrix(img, show=False)
        # shape could be (x, y) or (3, x, y)
        assert img.shape in [self.nn.input_shape[1:], self.nn.input_shape[2:]]
        return img.flatten()

    def run_only_last(self, img):
        img = self._prepare_img_to_run(img)
        return self.func([img])

    def predict(self, img):
        """ return predicted label (either a list or a digit)"""
        results = [self.run_only_last(img)]
        label = NetworkRunner.get_label_from_result(img, results)
        return label

    @staticmethod
    def get_label_from_result(img, results):
        """ parse the results and get label
            results: return value of run() or run_only_last()
        """
        return results[-1].argmax()

    def dump_params(self, filename):
        res = {}
        layers = self.nn.layers

        for layer, cnt in izip(layers, count()):
            # save layer type
            dic = {'type': cls_name_dict[type(layer)] }
            # save other layer parameters
            dic.update(layer.get_params())
            res['layer' + str(cnt)] = dic
        with gzip.open(filename, 'wb') as f:
            pickle.dump(res, f, -1)

def get_nlayer_from_params(params):
    for nlayer in count():
        layername = 'layer' + str(nlayer)
        if layername not in params:
            return nlayer

def build_nn_with_params(params, batch_size):
    """ build a network and return it
        params: the object load from param{epoch}.pkl.gz file
    """
    input_size = params['layer0']['input_shape']
    if batch_size is None:
        batch_size = input_size[0]
    input_size = (batch_size,) + input_size[1:]
    print "Size={0}".format(input_size)

    nlayer = get_nlayer_from_params(params)
    runner = NetworkRunner(input_size)

    if 'last_updates' in params:
        runner.set_last_updates(params['last_updates'])

    for idx in range(nlayer):
        layername = 'layer' + str(idx)
        layerdata = params[layername]
        typename = layerdata['type']
        if typename == 'convpool':
            typename = 'conv'
        layer_cls = name_cls_dict[typename]
        print "Layer ", idx, ' is ', layer_cls
        runner.nn.add_layer(layer_cls, layerdata)

    print "Model Loaded."
    return runner

def get_nn(filename, batch_size=1):
    """ get a network from a saved model file
        batch_size is None: will use same batch_size in the model
    """
    with gzip.open(filename, 'r') as f:
        data = pickle.load(f)

    nn = build_nn_with_params(data, batch_size)
    # compile all the functions
    nn.finish()
    #nn.nn.print_config()
    return nn
