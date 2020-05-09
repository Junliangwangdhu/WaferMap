# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
"""
Created on Wed Mar 28 09:52:58 2018

@author: xingshuli
"""

import numpy as np

#Map the input array to new coordinates by interpolation
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates

import tensorflow as tf

#flatten tensor
def tf_flatten(a):
    return tf.reshape(a, [-1])


#Tensorflow version of np.repeat for 1D
def tf_repeat(a, repeats, axis = 0):
    assert len(a.get_shape()) == 1
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    
    return a


#Tensorflow version of np.repeat for 2D
def tf_repeat_2d(a, repeats):
    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

#Tensorflow version of scipy.ndimage.map_coordinates
'''
Parameters:
input: tf.Tensor. shape = (s, s)
coords: tf.Tensor. shape = (n_points, 2)
coords_lt -- left-top of coordinates
coords_rb -- right-bottom of coordinates
coords_lb -- left-bottom of coordinates
coords_rt -- right-top of coordinates 

for mapped_vals is calculated by bilinear interpolation

'''
def tf_map_coordinates(input, coords, order = 1):
    assert order == 1 # '1' means the linear interpolation

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis = 1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis = 1)
    
    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)
    
    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    
    return mapped_vals

def sp_batch_map_coordinates(inputs, coords):
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([sp_map_coordinates(input, coord.T, mode = 'nearest', order = 1) 
                            for input, coord in zip(inputs, coords)])
    
    return mapped_vals
    
def tf_batch_map_coordinates(input, coords, order = 1):
    #Batch version of tf_map_coordinates
    '''
    Parameter
    input: tf.Tensor. shape = (b, s, s)
    coords: tf.Tensor. shape = (b, n_points, 2)
    
    Return
    tf. Tensor. shape = (b, s, s)
    '''
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]
    
    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)
    
    idx = tf_repeat(tf.range(batch_size), n_coords)
    
    def _get_vals_by_coords(input, coords):
        indices = tf.stack([idx, tf_flatten(coords[..., 0]), 
                            tf_flatten(coords[..., 1])], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)
    
    #bilinear interpolation
    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]
    
    return mapped_vals
    
def sp_batch_map_offsets(input, offsets):
    '''
    Reference implementation for tf_batch_map_offsets
    '''
    batch_size = input.shape[0]
    input_size = input.shape[1]
    
    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis = 0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)
    
    mapped_vals = sp_batch_map_coordinates(input, coords)
    
    return mapped_vals
    
def tf_batch_map_offsets(input, offsets, order = 1):
    '''
    Parameters:
    
    input: tf. Tensor. shape = (b, s, s)
    offsets: tf. Tensor. shape = (b, s, s, 2)
    
    Returns:
    tf. Tensor. shape = (b, s, s)
    
    '''
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    
    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(tf.range(input_size), tf.range(input_size), indexing = 'ij')
    grid = tf.stack(grid, axis = -1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = grid + offsets
    
    mapped_vals = tf_batch_map_coordinates(input, coords)
    
    return mapped_vals
    




