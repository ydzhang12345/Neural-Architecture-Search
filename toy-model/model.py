from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import math

import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops


def model_fn(actions, model_settings):
    fingerprint_input = tf.placeholder(tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')
    is_training = tf.placeholder(tf.bool, [])
    filters_1, filters_2, filters_3 = actions
    net = tf.layers.dense(fingerprint_input, filters_1, activation=tf.nn.relu)
    net = tf.layers.dense(net, filters_2, activation=tf.nn.relu)
    net = tf.layers.dense(net, filters_3, activation=tf.nn.relu)
    net = tf.layers.dense(net, model_settings['label_count'])
    return net, fingerprint_input, is_training




def model_fn_cnn(actions, model_settings):
    def ds_cnn_arg_scope(weight_decay=0):
        with slim.arg_scope(
            [slim.convolution2d, slim.separable_convolution2d],
            weights_initializer=slim.initializers.xavier_initializer(),
            biases_initializer=slim.init_ops.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc+'/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn


    fingerprint_input = tf.placeholder(tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')
    is_training = tf.placeholder(tf.bool, [])
    filters_1, kernel_1, stride_1, filters_2, kernel_2, stride_2, filters_3, kernel_3, stride_3 = actions

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
 
    t_dim = math.ceil(input_time_size / float(2 * stride_1 * stride_2 * stride_3))
    f_dim = math.ceil(input_frequency_size / float(1 * stride_1 * stride_2 * stride_3))


    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        outputs_collections=[end_points_collection]):
        with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          decay=0.96,
                          updates_collections=None,
                          activation_fn=tf.nn.relu):
            #pdb.set_trace()
            net = slim.convolution2d(fingerprint_4d, 64, [10, 4], stride=[2, 1], padding='SAME', scope='conv_1')
            net = slim.batch_norm(net, scope='conv_1/batch_norm')
            net = _depthwise_separable_conv(net, filters_1, kernel_size = [kernel_1, kernel_1], stride = [stride_1, stride_1], sc='conv_ds_1')
            net = _depthwise_separable_conv(net, filters_2, kernel_size = [kernel_2, kernel_2], stride = [stride_2, stride_2], sc='conv_ds_2')
            net = _depthwise_separable_conv(net, filters_3, kernel_size = [kernel_3, kernel_3], stride = [stride_3, stride_3], sc='conv_ds_3')
            net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')
    return logits, fingerprint_input, is_training

