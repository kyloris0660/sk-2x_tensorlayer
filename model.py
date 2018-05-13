import tensorflow as tf
import tensorlayer as tl
from config import *


def create_model(patches):
    with tf.variable_scope('vgg7'):
        net = tl.layers.InputLayer(patches, name='input_layer')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 3, 16),
                                    padding='VALID',
                                    name='vgg7_Conv1')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 16, 32),
                                    padding='VALID',
                                    name='vgg7_Conv2')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 32, 64),
                                    padding='VALID',
                                    name='vgg7_Conv3')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 64, 128),
                                    padding='VALID',
                                    name='vgg7_Conv4')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 128, 128),
                                    padding='VALID',
                                    name='vgg7_Conv5')
        net = tl.layers.Conv2dLayer(net, act=tf.nn.leaky_relu,
                                    shape=(3, 3, 128, 256),
                                    padding='VALID',
                                    name='vgg7_Conv6')

        batch_size = int(net.outputs.shape[0])
        rows = int(net.outputs.shape[1])
        cows = int(net.outputs.shape[2])
        channels = int(patches.get_shape()[0])

        net = tl.layers.DeConv2dLayer(net,
                                      shape=(4, 4, 3, 256),
                                      output_shape=(batch_size, rows * 2, cows * 2, channels),
                                      strides=(1, 2, 2, 1),
                                      name='vgg7_Deconv')
        return net


def s_mse_loss(inference, ground_truth, name='mse_loss'):
    with tf.name_scope(name):
        slice_begin = (int(ground_truth.get_shape()[1]) - int(inference.get_shape()[1])) // 2
        slice_end = int(inference.get_shape()[1]) + slice_begin
        delta = inference - ground_truth[:, slice_begin: slice_end, slice_begin: slice_end, :]

        delta *= [[[[0.11448, 0.58661, 0.29891]]]]  # weights of B, G and R
        l2_loss = tf.pow(delta, 2)
        mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=[1, 2, 3]))
        return mse_loss
