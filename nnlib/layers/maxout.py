#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: maxout.py
# Date: Wed Sep 17 16:00:18 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import theano.tensor as T
from common import Layer

class MaxoutLayer(Layer):
    NAME = 'maxout'

    def __init__(self, input_train, input_test,
                 input_shape, maxout_unit):
        super(MaxoutLayer, self).__init__(None, input_train, input_test)
        self.input_shape = input_shape
        self.maxout_unit = maxout_unit

        n_channel = input_shape[1]
        assert n_channel % maxout_unit == 0
        n_channel /= maxout_unit

        def do_maxout(input):
            output = input[:, ::maxout_unit]
            for i in range(1, maxout_unit):
                output = T.maximum(output, input[:, i::maxout_unit])
            return output

        self.output_train = do_maxout(input_train)
        if self.has_dropout_input:
            self.output_test = do_maxout(input_test)

    def get_params(self):
        return {'input_shape': self.input_shape,
                'maxout_unit': self.maxout_unit}

    def get_output_shape(self):
        return (self.input_shape[0], self.input_shape[1] / self.maxout_unit,
                self.input_shape[2], self.input_shape[3])

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = MaxoutLayer(input_train, input_test,
                            params['input_shape'],
                            params['maxout_unit'])
        return layer
