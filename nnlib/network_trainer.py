#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: network_trainer.py
from itertools import chain, izip
import operator
import pprint
import os
pprint = pprint.PrettyPrinter(indent=4).pprint

import numpy as np
import theano
import theano.tensor as T
import theano.printing as PP

from layers.layers import *

N_OUT = 10
MOMENT = 0.6

class NNTrainer(object):
    """ Neural Network Trainer
    """
    def __init__(self, input_image_shape, multi_output=True):
        """ input_image_shape: 4D tuple
            multi_output: whether a image has more than 1 labels
        """
        self.layer_config = []
        self.layers = []
        self.batch_size = input_image_shape[0]
        self.input_shape = input_image_shape
        self.rng = np.random.RandomState()
        self.multi_output = multi_output

        self.x = T.fmatrix('x')
        Layer.x = self.x        # only for debug purpose
        if multi_output:
            self.y = T.imatrix('y')
        else:
            self.y = T.ivector('y')

        self.orig_input = self.x.reshape(self.input_shape)
        self.last_updates = []

    def add_layer(self, layer_class, params):
        """ add a layer to the network.
            layer_class: the layer class,
            params: a dict with necessary params.
            input_shape is not needed in params, since it will be derived from the previous layer.
        """
        assert issubclass(layer_class, Layer)
        if len(self.layers) == 0:
            # first layer
            params['input_shape'] = self.input_shape
            print params['input_shape']
            layer = layer_class.build_layer_from_params(
                params, self.rng, self.orig_input)
        else:
            last_layer = self.layers[-1]
            params['input_shape'] = last_layer.get_output_shape()
            print params['input_shape']
            layer = layer_class.build_layer_from_params(
                params, self.rng,
                last_layer.get_output_train(),
                last_layer.get_output_test())
        self.layers.append(layer)
        params['type'] = layer_class.get_class_name()

        # remove W & b in params, for better printing
        params = dict([k, v] for k, v in params.iteritems() if type(v) not in [np.ndarray, list])
        self.layer_config.append(params)

    def n_params(self):
        """ Calculate total number of params in this model"""
        def get_layer_nparam(layer):
            prms = layer.params
            ret = sum([reduce(operator.mul, k.get_value().shape) for k in prms])
            if ret > 0:
                print "Layer {0} has {1} params".format(type(layer), ret)
            return ret
        return sum([get_layer_nparam(l) for l in self.layers])

    def print_config(self):
        print "Network has {0} params in total.".format(self.n_params())
        pprint(self.layer_config)

    def finish(self):
        """ call me before training"""
        self.print_config()

        if type(self.layers[-1]) == SequenceSoftmax:
            self.max_len = self.layer_config[-1]['seq_max_len']
            print "Using Sequence Softmax Output with max_len = {0}".format(self.max_len)
        else:
            self.max_len = 0

        layer = self.layers[-1]
        assert type(layer) in [SequenceSoftmax, LogisticRegression]
        # cost to minimize
        self.cost = layer.negative_log_likelihood(self.y)

        # all the params to optimize on
        self.params = list(chain.from_iterable([x.params for x in self.layers]))

        # take derivatives on those params
        self.grads = T.grad(self.cost, self.params)

        # save last updates for momentum
        if not self.last_updates:
            self.last_updates = []
            for param in self.params:
                self.last_updates.append(
                    theano.shared(
                        np.zeros(param.get_value(borrow=True).shape,
                                 dtype=theano.config.floatX)
                    ))
        assert len(self.last_updates) == len(self.params), 'last updates don\'t match params'
