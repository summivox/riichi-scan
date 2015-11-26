#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: imageutil.py
# Date: Thu Dec 04 20:45:54 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import cv2
import numpy
import numpy as np
from itertools import izip

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def get_image_matrix(img, show=True):
    """ if img is flattened, make it a matrix
        show: if img is (3, x, y), make it (x, y, 3)
        not show: make img (3, x, y) for run
    """
    shape = img.shape
    if len(shape) == 2:
        return img
    if len(shape) == 3:
        if shape[0] != 3:
            assert shape[2] == 3
            if show:
                return img
            else:
                ret = numpy.asarray((img[..., 0], img[..., 1], img[..., 2]))
                return ret
        if not show:
            return img
        ret = numpy.zeros((shape[1], shape[2], 3))
        ret[..., 0] = img[0]
        ret[..., 1] = img[1]
        ret[..., 2] = img[2]
        return ret
    l = int(numpy.sqrt(shape[0]))
    assert l * l == int(shape[0])
    return img.reshape((l, l))

import matplotlib.pyplot as plot
def show_img_sync(img):
    """ synchronous function to display a image"""
    k = get_image_matrix(img)
    plot.imshow(k)
    plot.show()


def get_label_from_dataset(dt, label):
    """ get a digit with certain label
        dt: tuple of (imgs, labels)
    """
    for img, l in izip(dt[0], dt[1]):
        if label == l:
            return get_image_matrix(img)

def stack_vectors(vecs):
    """stack a list of vectors, in order to show them in color"""
    image_thickness = 50
    ret = np.vstack([vecs[0]] * image_thickness)
    shape = vecs[0].shape
    for k in range(1, len(vecs)):
        ret = np.vstack((ret, [np.zeros(shape)] * 5, [vecs[k]] * image_thickness))
    return ret

def padding(img, shape, fill):
    h, w = img.shape[:2]
    assert w <= shape[0] and h <= shape[1]
    pad_width = shape[0] - w
    pad_height = shape[1] - h

    pad_w0 = pad_width / 2
    pad_w1 = shape[0] - (pad_width - pad_w0)
    pad_h0 = pad_height / 2
    pad_h1 = shape[1] - (pad_height - pad_h0)

    ret = np.ones((shape[1], shape[0]), dtype='uint8') * fill
    ret[pad_h0:pad_h1,pad_w0:pad_w1] = img
    return ret

def resize_preserve(img, shape, fill_empty=0):
    h, w = img.shape[:2]
    tw, th = shape[:2]

    w_ratio = tw / float(w)
    h_ratio = th / float(h)

    ratio = min(w_ratio, h_ratio)

    cw = int(max(1, w * ratio))
    ch = int(max(1, h * ratio))
    img = cv2.resize(img, (cw, ch))
    return padding(img, shape, fill_empty)
