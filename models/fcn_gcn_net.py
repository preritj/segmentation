import tensorflow as tf
from layers.layers_fcn_gcn import (
    conv_module, global_conv_module, boundary_refine, deconv_module)
from models.base_model import SegModel


class FCNGCNnet(SegModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def preprocess(self, inputs):
        """Image preprocessing"""
        h, w = self.cfg.input_shape
        inputs = tf.reshape(inputs, [-1, h, w, 3])
        return 2.0 * tf.to_float(inputs) / 255. - 1.0

    def build_net(self, input_, is_training=False):
        """Based on https://arxiv.org/abs/1703.02719 but using VGG style base
        Args:
            input_ (4-D Tensor): (N, H, W, C)
            is_training (bool): If True, run in training mode
        Returns:
            output (4-D Tensor): (N, H, W, n)
                Logits classifying each pixel as either 'car' (1) or 'not car' (0)
        """
        num_classes = self.cfg.num_classes  # Number of classes
        k_gcn = self.cfg.k_gcn  # Kernel size for global conv layer
        init_channels = self.cfg.init_channels  # Number of channels in the first conv layer
        n_layers = self.cfg.n_layers  # Number of times to downsample/upsample
        batch_norm = self.cfg.batch_norm  # if True, use batch-norm

        # color-space adjustment
        net = tf.layers.conv2d(input_, 3, (1, 1), name="color_space_adjust")
        n = n_layers

        # encoder
        feed = net
        ch = init_channels
        conv_blocks = []
        for i in range(n-1):
            conv, feed = conv_module(feed, ch, is_training, name=str(i + 1),
                                     batch_norm=batch_norm)
            conv_blocks.append(conv)
            ch *= 2
        last_conv = conv_module(feed, ch, is_training, name=str(n), pool=False,
                                batch_norm=batch_norm)
        conv_blocks.append(last_conv)

        # global convolution network
        global_conv_blocks = []
        for i in range(n):
            global_conv_blocks.append(
                global_conv_module(conv_blocks[i], num_classes, is_training,
                                   k=k_gcn, name=str(i + 1)))

        # boundary refinement
        br_blocks = []
        for i in range(n):
            br_blocks.append(boundary_refine(global_conv_blocks[i], is_training,
                                             name=str(i + 1), batch_norm=batch_norm))

        # decoder / upsampling
        up_blocks = []
        last_br = br_blocks[-1]
        for i in range(n-1, 0, -1):
            deconv = deconv_module(last_br, name=str(i+1), stride=2, kernel_size=4)
            up = tf.add(deconv, br_blocks[i - 1])
            last_br = boundary_refine(up, is_training, name='up_' + str(i))
            up_blocks.append(up)

        logits = last_br
        return logits

