import tensorflow as tf
from layers_fcn_gcn import conv_module, global_conv_module, boundary_refine, deconv_module


def fcn_gcn_net(input_, num_classes, k_gcn, training, init_channels=8, n_layers=7, batch_norm=True):
    """Based on https://arxiv.org/abs/1703.02719 but using VGG style base
    Args:
        input_ (4-D Tensor): (N, H, W, C)
        num_classes (integer) : Number of classes
        k_gcn (int) : Kernel size for global conv layer
        training (bool): If True, run in training mode
        init_channels (int) : Number of channels in the first conv layer
        n_layers (int) : Number of times to downsample/upsample
        batch_norm (bool): if True, use batch-norm
    Returns:
        output (4-D Tensor): (N, H, W, n)
            Logits classifying each pixel as either 'car' (1) or 'not car' (0)
    """
    # color-space adjustment
    net = tf.layers.conv2d(input_, 3, (1, 1), name="color_space_adjust")
    n = n_layers

    # encoder
    feed = net
    ch = init_channels
    conv_blocks = []
    for i in range(n-1):
        conv, feed = conv_module(feed, ch, training, name=str(i + 1), batch_norm=batch_norm)
        conv_blocks.append(conv)
        ch *= 2
    last_conv = conv_module(feed, ch, training, name=str(n), pool=False, batch_norm=batch_norm)
    conv_blocks.append(last_conv)

    # global convolution network
    global_conv_blocks = []
    for i in range(n):
        global_conv_blocks.append(global_conv_module(conv_blocks[i], num_classes, training,
                                                    k = k_gcn, name=str(i + 1)))

    # boundary refinement
    br_blocks = []
    for i in range(n):
        br_blocks.append(boundary_refine(global_conv_blocks[i], training, name=str(i + 1),
                                         batch_norm=batch_norm))

    # decoder / upsampling
    up_blocks = []
    last_br = br_blocks[-1]
    for i in range(n-1, 0, -1):
        deconv = deconv_module(last_br, name=str(i+1), stride=2, kernel_size=4)
        up = tf.add(deconv, br_blocks[i - 1])
        last_br = boundary_refine(up, training, name='up_' + str(i))
        up_blocks.append(up)

    logits = last_br
    return logits

