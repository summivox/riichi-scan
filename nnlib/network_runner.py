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
    def __init__(self, input_shape, multi_output):
        """ input size in (height, width)"""
        # nn is the underlying neural network object to run with
        self.nn = NNTrainer(input_shape, multi_output)
        self.multi_output = multi_output

    def get_layer_by_index(self, idx):
        """ return the instance of certain layer.
            idx can be negative to get layers from the end
        """
        return self.nn.layers[idx]

    def set_last_updates(self, last_updates):
        """ set last_updates in trainer, for used in momentum
            last_updates: list of np array for each param
        """
        assert len(self.nn.last_updates) == 0
        for lu in last_updates:
            self.nn.last_updates.append(theano.shared(lu))

    def finish(self, only_last=True):
        """ compile the output of each layer as theano function"""
        print "Compiling..."
        self.funcs = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx != len(self.nn.layers) - 1 and only_last:
                continue
            if idx == len(self.nn.layers) - 1:
                # the output layer: use likelihood of the label
                f = theano.function([self.nn.x],
                                     layer.p_y_given_x,
                                    allow_input_downcast=True)
            else:
                # layers in the middle: use its output fed into the next layer
                f = theano.function([self.nn.x],
                                   layer.get_output_test(), allow_input_downcast=True)
            self.funcs.append(f)

    def _prepare_img_to_run(self, img):
        assert self.nn.batch_size == 1, \
                "batch_size of runner is not 1, but trying to run against 1 image"
        img = get_image_matrix(img, show=False)
        # shape could be (x, y) or (3, x, y)
        assert img.shape in [self.nn.input_shape[1:], self.nn.input_shape[2:]]
        return img.flatten()

    def run(self, img):
        """ return all the representations after each layer"""
        img = self._prepare_img_to_run(img)
        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            # why [img]?
            # theano needs arguments to be listed, although there is only 1 argument here
            results.append(self.funcs[idx]([img]))
        return results

    def run_only_last(self, img):
        img = self._prepare_img_to_run(img)
        return self.funcs[-1]([img])

    def predict(self, img):
        """ return predicted label (either a list or a digit)"""
        results = [self.run_only_last(img)]
        label = NetworkRunner.get_label_from_result(img, results,
                                                    self.multi_output)
        return label

    @staticmethod
    def get_label_from_result(img, results, multi_output):
        """ parse the results and get label
            results: return value of run() or run_only_last()
        """
        if not multi_output:
            # the predicted results for single digit output
            return results[-1].argmax()
        else:
            # predicted results for multiple digit output
            ret = []
            for r in results[-1]:
                ret.append(r[0].argmax())
            return ret

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
    last_layer = params['layer{0}'.format(nlayer - 1)]
    if last_layer['type'] in ['ssm']:
        multi_output = True
    elif last_layer['type'] in ['lr']:
        multi_output = False
    else:
        assert False
    runner = NetworkRunner(input_size, multi_output)

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
    nn.nn.print_config()
    return nn

#def save_LR_W_img(W, n_filter):
    #""" save W as images """
    #for l in range(N_OUT):
        #w = W[:,l]
        #size = int(np.sqrt(w.shape[0] / n_filter))
        #imgs = w.reshape(n_filter, size, size)
        #for idx, img in enumerate(imgs):
            #imsave('LRW-label{0}-weight{1}.jpg'.format(l, idx), img)

#def save_convolved_images(nn, results):
    #for nl in xrange(nn.n_conv_layer):
        #layer = results[nl][0]
        #img_shape = layer[0].shape
        #tile_len = int(np.ceil(np.sqrt(len(layer))))
        #tile_shape = (tile_len, int(np.ceil(len(layer) * 1.0 / tile_len)))
        #layer = layer.reshape((layer.shape[0], -1))
        #raster = tile_raster_images(layer, img_shape, tile_shape,
                                    #tile_spacing=(3, 3))
        #imsave('{0}.jpg'.format(nl), raster)

