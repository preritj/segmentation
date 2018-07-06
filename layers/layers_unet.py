import tensorflow as tf
import numpy as np
import math


def conv_module(input_, n_filters, training, name, pool=True, activation=tf.nn.relu,
                padding='same', batch_norm=True):
    """{Conv -> BN -> RELU} x 2 -> {Pool, optional}
        reference : https://github.com/kkweon/UNet-in-Tensorflow
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (int): depth of output tensor
        training (bool): If True, run in training mode
        name (str): name postfix
        pool (bool): If True, MaxPool2D after last conv layer
        activation: Activaion functions
        padding (str): 'same' or 'valid'
        batch_norm (bool) : If True, use batch-norm
    Returns:
        u_net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    kernel_sizes = [3,3]
    net = input_
    with tf.variable_scope("conv_module_{}".format(name)):
        for i, k_size in enumerate(kernel_sizes):
            net = tf.layers.conv2d(net, n_filters, (k_size, k_size), activation=None, padding=padding,
                                   name="conv_{}".format(i + 1))

            if batch_norm:
                net = tf.layers.batch_normalization(net, training=training, renorm=True,
                                                    name="bn_{}".format(i + 1))
            net = activation(net, name="relu_{}".format(i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool")

        return net, pool


def upsample(input_, name, upscale_factor=(2,2)):
    H, W, _ = input_.get_shape().as_list()[1:]

    target_H = H * upscale_factor[0]
    target_W = W * upscale_factor[1]

    return tf.image.resize_nearest_neighbor(input_, (target_H, target_W), name="upsample_{}".format(name))