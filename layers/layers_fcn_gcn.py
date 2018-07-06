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


def global_conv_module(input_, num_classes, training, name, k=13, padding='same'):
    """Global convolution network [https://arxiv.org/abs/1703.02719]
    Args:
         input_ (4-D Tensor): (batch_size, H, W, C)
         num_classes (integer) : Number of classes to classify
         name (str): name postfix
         k (integer): filter size for 1 x k + k x 1 convolutions
         padding (str) : 'same' or 'valid
    Returns:
         net (4-D Tensor): (batch_size, H, W, num_classes)
    """
    net = input_
    n_filters = num_classes

    with tf.variable_scope("global_conv_module_{}".format(name)):
        branch_a = tf.layers.conv2d(net, n_filters, (k, 1), activation=None,
                                     padding=padding, name='conv_1a')
        branch_a = tf.layers.conv2d(branch_a, n_filters, (1, k), activation=None,
                                     padding=padding, name='conv_2a')

        branch_b = tf.layers.conv2d(net, n_filters, (1, k), activation=None,
                                     padding=padding, name='conv_1b')
        branch_b = tf.layers.conv2d(branch_b, n_filters, (k, 1), activation=None,
                                     padding=padding, name='conv_2b')

        net = tf.add(branch_a, branch_b, name='sum')

        return net


def boundary_refine(input_, training, name, activation=tf.nn.relu, batch_norm=True):
    """Boundary refinement network [https://arxiv.org/abs/1703.02719]
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        training (bool): If True, run in training mode
        name (str): name postfix
        activation: Activaion functions
        batch_norm (bool) : Whether to use batch norm
    Returns:
        net (4-D Tensor): output tensor of same shape as input_
    """
    net = input_
    n_filters = input_.get_shape()[3].value

    with tf.variable_scope("boundary_refine_module_{}".format(name)):

        net = tf.layers.conv2d(net, n_filters, (3, 3), activation=None,
                               padding='SAME', name='conv_1')
        if batch_norm:
            net = tf.layers.batch_normalization(net, training=training,
                                                name='bn_1', renorm=True)
        net = activation(net, name='relu_1')

        net = tf.layers.conv2d(net, n_filters, (3, 3), activation=None,
                               padding='SAME', name='conv_2')
        net = tf.add(net, input_, name='sum')

        return net


def get_deconv_filter(name, n_channels, k_size):
    """Creates weight kernel initialization for deconvolution layer
        reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    Args:
        name (str): name postfix
        n_channels (int): number of input and output channels are same
        k_size (int): kernel-size (~ 2 x stride for FCN case)
    Returns:
        weight kernels (4-D Tensor): (k_size , k_size, n_channels, n_channels)
    """
    k = k_size
    filter_shape = [k, k, n_channels, n_channels]
    f = math.ceil(k / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros((k, k))
    for x, y in zip(range(k), range(k)):
        bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    weights = np.zeros(filter_shape)
    for i in range(n_channels):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def deconv_module(input_, name, stride=2, kernel_size=4, padding='SAME'):
    """ Convolutional transpose layer for upsampling score layer
        reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        name (str): name postfix
        stride (int): the upscaling factor (default is 2)
        kernel_size (int): (~ 2 x stride for FCN case)
        padding (str): 'same' or 'valid'
    Returns:
        net: output of transpose convolution operations
    """
    n_channels = input_.get_shape()[3].value
    strides = [1, stride, stride, 1]
    in_shape = input_.get_shape()
    h = in_shape[1].value * stride
    w = in_shape[2].value * stride
    out_shape = tf.stack([in_shape[0].value, h, w, n_channels])
    with tf.variable_scope('deconv_{}'.format(name)):
        weights = get_deconv_filter('up_filter_kernel', n_channels, k_size=kernel_size)
        deconv = tf.nn.conv2d_transpose(input_, weights, output_shape=out_shape,
                                        strides=strides, padding=padding)
        return deconv
