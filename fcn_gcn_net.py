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


def dice_coef(y_true, y_pred, axis=None, smooth = 0.001):
    """ Calculates dice coefficient in tensorflow :
     https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Args:
        y_true (Tensor): ground truth masks
        y_pred (Tensor): predicted masks
        axis : dimensions along which dice coeff is calculated (default is [1,2])
        smooth : a small number added to prevent divide by zero
    Returns:
        dice coefficient
    """
    if axis is None:
        axis=[1,2]
    y_true_f = tf.cast(y_true, dtype=tf.float32)
    y_pred_f = tf.cast(y_pred, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=axis)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=axis)
                                           + tf.reduce_sum(y_pred_f, axis=axis) + smooth)
    return tf.reduce_mean(dice)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def pixel_wise_loss(pixel_logits, gt_pixels, pixel_weights=None):
    """Calculates pixel-wise softmax cross entropy loss
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
        gt_pixels (3-D Tensor): Image masks of shape (N, H, W)
        pixel_weights (3-D Tensor) : (N, H, W) Weights for each pixel
    Returns:
        scalar loss : softmax cross-entropy
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pixel_logits, labels=gt_pixels)
    if pixel_weights is None:
        return tf.reduce_mean(loss)
    else:
        return tf.reduce_sum(loss * pixel_weights) / tf.reduce_sum(pixel_weights)


def mask_prediction(pixel_logits):
    """
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
    Returns:
        Predicted pixel-wise probabilities (3-D Tensor): (N, H, W)
        Predicted mask (3-D Tensor): (N, H, W)
    """
    probs = tf.nn.softmax(pixel_logits)
    n, h, w, _ = probs.get_shape()
    masks = tf.reshape(probs, [-1, 2])
    masks = tf.argmax(masks, axis=1)
    masks = tf.reshape(masks, [n.value, h.value, w.value])
    return probs, masks
