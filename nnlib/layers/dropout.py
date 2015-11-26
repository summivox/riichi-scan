#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dropout.py
# Date: Thu Sep 18 04:07:32 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from common import Layer

class DropoutLayer(Layer):
    NAME = 'dropout'
    def __init__(self, rng, input_train, input_test,
                 input_shape, dropout):
        super(DropoutLayer, self).__init__(rng, input_train, input_test)
        self.input_shape = input_shape
        self.dropout = dropout

        self.output_train = self.input_train
        if self.has_dropout_input:
            self.output_test = self.input_test

    def get_output_shape(self):
        return self.input_shape

    def get_params(self):
        return {'input_shape': self.input_shape,
                'dropout': self.dropout}

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = DropoutLayer(rng, input_train, input_test,
                             params['input_shape'],
                             params.get('dropout', 0.5))
        return layer
