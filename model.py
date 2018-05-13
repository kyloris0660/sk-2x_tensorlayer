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

