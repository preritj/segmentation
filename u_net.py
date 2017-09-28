import tensorflow as tf
from layers_unet import conv_module, upsample


def unet(input_, num_classes, training, init_channels=8, n_layers=6,  batch_norm=True):
    """Based on https://arxiv.org/abs/1505.04597
    Args:
        input_ (4-D Tensor): (N, H, W, C)
        num_classes (int) : Number of classes
        n_layers (int) : Number of times to downsample/upsample
        training (bool): If True, run in training mode
        init_channels (int) : Number of channels in the first conv layer
        batch_norm (bool): if True, use batch-norm
    Returns:
        output (4-D Tensor): (N, H, W, n)
            Logits classifying each pixel as either 'car' (1) or 'not car' (0)
    """
    # color-space adjustment
    net = tf.layers.conv2d(input_, 3, (1, 1), name="color_space_adjust")

    # encoder
    feed = net
    ch = init_channels
    conv_blocks = []
    for i in range(n_layers):
        conv, feed = conv_module(feed, ch, training, name='down_{}'.format(i + 1), batch_norm=batch_norm)
        conv_blocks.append(conv)
        ch *= 2
    last_conv = conv_module(feed, ch, training, name='down_{}'.format(n_layers+1),
                            pool=False, batch_norm=batch_norm)
    conv_blocks.append(last_conv)

    # decoder / upsampling
    feed = conv_blocks[-1]
    for i in range(n_layers, 0, -1):
        ch /= 2
        up = upsample(feed, name=str(i+1))
        concat = tf.concat([up, conv_blocks[i-1]], axis=-1, name="concat_{}".format(i))
        feed = conv_module(concat, ch, training, name='up_{}'.format(i), batch_norm=batch_norm,
                           pool=False)

    logits = tf.layers.conv2d(feed, num_classes, (1, 1), name='logits', activation=None, padding='same')
    return logits

