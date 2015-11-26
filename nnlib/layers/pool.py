#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Date: Thu Sep 18 01:55:04 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import __builtin__
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import theano.printing as PP
import numpy as np

from common import Layer

class PoolLayer(Layer):
    NAME = 'pool'

    def __init__(self, input_train, input_test,
                 image_shape, pool_size, stride):
        super(PoolLayer, self).__init__(None, input_train, input_test)
        self.image_shape = image_shape
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size
        self.stride = stride

        if stride is None:
            def gen_pool(input):
                return downsample.max_pool_2d(input=input,
                                                    ds=pool_size,
                                                    ignore_border=True)
        else:
            assert stride == 1
            assert pool_size == (2, 2)
            def gen_pool(input):
                ret = T.zeros_like(input)
                m00 = downsample.max_pool_2d(input, pool_size, True)
                #print "m", m00.shape.eval({Layer.x:np.random.rand(64, 3*64*64).astype('float32')})
                m01 = downsample.max_pool_2d(input[:,:,:,1:], pool_size, True)
                m10 = downsample.max_pool_2d(input[:,:,1:,:], pool_size, True)
                m11 = downsample.max_pool_2d(input[:,:,1:,1:], pool_size, True)
                ret = T.set_subtensor(ret[:,:,::2,::2], m00)
                ret = T.set_subtensor(ret[:,:,::2,1:-1:2], m01)
                ret = T.set_subtensor(ret[:,:,1:-1:2,::2], m10)
                ret = T.set_subtensor(ret[:,:,1:-1:2,1:-1:2], m11)
                # set the last row
                row0 = downsample.max_pool_2d(input[:,:,-1,:].dimshuffle(0, 1, 'x', 2), (1, 2), True)
                ret = T.set_subtensor(ret[:,:,-1,::2], row0[:,:,0,:])
                row1 = downsample.max_pool_2d(input[:,:,-1,1:].dimshuffle(0, 1, 'x', 2), (1, 2), True)
                ret = T.set_subtensor(ret[:,:,-1,1:-1:2], row1[:,:,0,:])
                # set the last column
                col0 = downsample.max_pool_2d(input[:,:,:,-1].dimshuffle(0, 1, 2, 'x'), (2, 1), True)
                ret = T.set_subtensor(ret[:,:,::2,-1], col0[:,:,:,0])
                col1 = downsample.max_pool_2d(input[:,:,1:,-1].dimshuffle(0, 1, 2, 'x'), (2, 1), True)
                ret = T.set_subtensor(ret[:,:,1:-1:2,-1], col1[:,:,:,0])
                # set the last element
                ret = T.set_subtensor(ret[:,:,-1,-1], input[:,:,-1,-1])

                # wrong here
                #ret = T.set_subtensor(ret[:,:,-1:,:], input[:,:,-1,:])
                #ret = T.set_subtensor(ret[:,:,:,-1], input[:,:,:,-1])
                return ret

        self.output_train = gen_pool(self.input_train)
        if self.has_dropout_input:
            self.output_test = gen_pool(self.input_test)

    def get_output_shape(self):
        if self.stride is None:
            return (self.image_shape[0], self.image_shape[1],
                    self.image_shape[2] / self.pool_size[0],
                    self.image_shape[3] / self.pool_size[1])
        else:
            return (self.image_shape[0], self.image_shape[1],
                    self.image_shape[2] / self.stride,
                    self.image_shape[2] / self.stride)

    def get_params(self):
        return {'pool_size': self.pool_size,
                'input_shape': self.image_shape,
                'stride': self.stride
               }

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = PoolLayer(input_train, input_test,
                          params['input_shape'],
                          params['pool_size'],
                          params.get('stride', None))
        return layer

