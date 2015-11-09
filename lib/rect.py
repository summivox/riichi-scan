#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: rect.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np

class Rect(object):
    __slots__ = ['x', 'y', 'w', 'h']

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        assert min(self.x, self.y, self.w, self.h) >= 0

    @property
    def x0(self):
        return self.x

    @property
    def y0(self):
        return self.y

    @property
    def x1(self):
        return self.x + self.w - 1

    @property
    def y1(self):
        return self.y + self.h - 1

    def copy(self):
        new = type(self)()
        for i in self.__slots__:
            setattr(new, i, getattr(self, i))
        return new

    def slice_ndarray(self, arr):
        assert all(isinstance(getattr(self, i), int)
                   for i in self.__slots__)
        return arr[self.y0:self.y1, self.x0:self.x1]

    def __str__(self):
        return 'Rect(x={}, y={}, w={}, h={})'.format(
            self.x, self.y, self.w, self.h)

    def area(self):
        return self.w * self.h

    def intersects(self, rect):
        return not (self.x > rect.x1 or
                self.x1 < rect.x or
                self.y > rect.y1 or
                self.y1 < rect.y)

    def intersect_area(self, rect):
        if not self.intersects(rect):
            return 0
        return min(rect.x1 - self.x, self.x1 - rect.x) * min(rect.y1 - self.y, self.y1 - rect.y)

    def intersect_ratio(self, rect):
        area = self.intersect_area(rect)
        min_area = min([self.area(), rect.area()])
        return area * 1.0 / min_area

    __repr__ = __str__
