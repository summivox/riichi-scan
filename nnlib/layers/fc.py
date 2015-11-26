#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fc.py
# Date: Wed Sep 17 16:23:23 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import theano
import theano.tensor as T
import theano.printing as PP
import scipy.io as sio

from common import ReLu, Layer

class FullyConnectedLayer(Layer):
    NAME = 'fc'

    def __init__(self, rng, input_train, input_test,
                 input_shape, n_out,
                 activation):
        super(FullyConnectedLayer, self).__init__(rng, input_train, input_test)
        self.input_shape = input_shape
        n_in = np.prod(input_shape[1:])
        self.n_out = n_out
        self.activation = activation

        W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == T.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        def do_cal(input):
            input = input.reshape((input_shape[0], n_in))
            linear_output = T.dot(input, self.W) + self.b
            output = (linear_output if activation is None else activation(linear_output))
            return output

        self.output_train = do_cal(self.input_train)
        if self.has_dropout_input:
            self.output_test = do_cal(self.input_test)
        else:
            self.output_test = self.output_train
        self.params = [self.W, self.b]

    def get_output_shape(self):
        return (self.input_shape[0], self.n_out)

    def get_params(self):
        return {'input_shape': self.input_shape,
                'n_out': self.n_out,
                'activation': self.activation,
                'W': self.W.get_value(borrow=True),
                'b': self.b.get_value(borrow=True)}

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        params = self.get_params()
        sio.savemat(basename + '.mat', params)

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = FullyConnectedLayer(rng, input_train, input_test,
                                    params['input_shape'],
                                    params['n_out'],
                                    params.get('activation', ReLu)
                                   )
        if 'W' in params:
            layer.W.set_value(params['W'].astype('float32'))
            layer.b.set_value(params['b'].astype('float32'))
        return layer
