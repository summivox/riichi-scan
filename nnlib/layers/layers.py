#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: layers.py
# Date: Wed Sep 17 16:16:41 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


#from logistic_sgd import LogisticRegression
#from fixed_length_softmax import FixedLengthSoftmax

from common import Layer

from conv import ConvLayer
from pool import PoolLayer
from sub import MeanSubtractLayer
from maxout import MaxoutLayer
from fc import FullyConnectedLayer
from LR import LogisticRegression
from dropout import DropoutLayer

layer_types = [ConvLayer, FullyConnectedLayer, PoolLayer,
               MeanSubtractLayer, MaxoutLayer, LogisticRegression,
               DropoutLayer]

cls_name_dict = dict([(k, k.get_class_name()) for k in layer_types])
name_cls_dict = dict([(v, k) for k, v in cls_name_dict.iteritems()])
