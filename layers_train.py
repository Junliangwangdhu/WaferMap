# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

"""
Created on Tue Apr 17 11:43:29 2018

@author: xingshuli
"""

import tensorflow as tf
from keras.layers import Conv2D
from keras.initializers import RandomNormal
from .deform_conv import tf_batch_map_offsets


class ConvOffset2D_train(Conv2D):
    '''
    Convolutional layer responsible for learning the 2D offsets and output the deformed
    feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature map

    '''

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        '''
        Parameters:
        filters: int
        Number of channel of the input feature map
        init_normal_stddev: float
        Normal kernel initialization
        **kwargs:
        pass to superclass. see the Conv2D layer in keras
        '''
        self.filters = filters
        #super(ConvOffset2D_test, self).__init__(self.filters, **kwargs)
        super(ConvOffset2D_train, self).__init__(self.filters * 2, (3, 3), padding = 'same',use_bias = False, kernel_initializer = RandomNormal(0, init_normal_stddev), **kwargs)

    def call(self, x):
        '''
        return the deformed featureed map
        '''
        x_shape = x.get_shape()
        offsets = super(ConvOffset2D_train, self).call(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)
        # X_offset: (b*c, h, w)
        x_offset = tf_batch_map_offsets(x, offsets)
        # x_offset: (b, h, w, c)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        '''
        Output shape is the same as input shape
        Becauase, this layer only does the deformation part
        '''
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        '''
        (b, h, w, 2c)->(bc, h, w, 2)
        '''
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        '''
        (b, h, w, c)->(bc, h, w)
        '''
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        '''
        (b*c, h, w)->(b, h, w, c)
        '''
        x = tf.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        x = tf.transpose(x, [0, 2, 3, 1])
        return x











