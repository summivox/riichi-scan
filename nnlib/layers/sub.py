#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sub.py
# Date: Wed Sep 17 16:00:26 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import theano
import numpy
from theano.tensor.nnet import conv

from common import Layer

def mean_filter(kernel_size):
    s = kernel_size ** 2
    x = numpy.repeat(1./s, s).reshape((kernel_size, kernel_size))
    return x

class MeanSubtractLayer(Layer):
    NAME = 'sub'

    def __init__(self, input_train, input_test,
                 input_shape, filter_size):
        super(MeanSubtractLayer, self).__init__(None, input_train, input_test)
        self.input_shape = input_shape
        self.filter_size = filter_size

        filter_shape = (1, 1, filter_size, filter_size)
        filters = mean_filter(filter_size).reshape(filter_shape)
        filters = theano.shared(numpy.asarray(filters,
                                              dtype=theano.config.floatX),
                                borrow=True)

        def do_sub(input):
            input = input.reshape((input_shape[0] * input_shape[1],
                                   1,
                                   input_shape[2], input_shape[3]))
            mean = conv.conv2d(input, filters=filters,
                               filter_shape=filter_shape, border_mode='full')
            mid = int(numpy.floor(filter_size / 2.))
            output = input - mean[:, :, mid : -mid, mid : -mid]
            output = output.reshape(self.input_shape)
            return output

        self.output_train = do_sub(input_train)
        if self.has_dropout_input:
            self.output_test = do_sub(input_test)

    def get_output_shape(self):
        return self.input_shape

    def get_params(self):
        return {'input_shape': self.input_shape,
                'filter_size': self.filter_size}

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        return MeanSubtractLayer(input_train, input_test,
                                 params['input_shape'],
                                 params['filter_size'])
