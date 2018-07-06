import tensorflow as tf
from layers.layers_unet import conv_module, upsample
from models.base_model import SegModel


class UNet(SegModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def preprocess(self, inputs):
        """Image preprocessing"""
        h, w = self.cfg.input_shape
        inputs = tf.reshape(inputs, [-1, h, w, 3])
        return 2.0 * tf.to_float(inputs) / 255. - 1.0

    def build_net(self, input_, is_training=False):
        """Based on https://arxiv.org/abs/1505.04597
        Args:
            input_ (4-D Tensor): (N, H, W, C)
            is_training (bool): If True, run in training mode
        Returns:
            output (4-D Tensor): (N, H, W, n)
                Logits classifying each pixel as either 'car' (1) or 'not car' (0)
        """
        num_classes = self.cfg.num_classes  # Number of classes
        n_layers = self.cfg.n_layers  # Number of times to downsample/upsample
        init_channels = self.cfg.init_channels  # Number of channels in the first conv layer
        batch_norm = self.cfg.batch_norm  # if True, use batch-norm

        # color-space adjustment
        net = tf.layers.conv2d(input_, 3, (1, 1), name="color_space_adjust")

        # encoder
        feed = net
        ch = init_channels
        conv_blocks = []
        for i in range(n_layers):
            conv, feed = conv_module(feed, ch, is_training, name='down_{}'.format(i + 1),
                                     batch_norm=batch_norm)
            conv_blocks.append(conv)
            ch *= 2
        last_conv = conv_module(feed, ch, is_training, name='down_{}'.format(n_layers+1),
                                pool=False, batch_norm=batch_norm)
        conv_blocks.append(last_conv)

        # decoder / upsampling
        feed = conv_blocks[-1]
        for i in range(n_layers, 0, -1):
            ch /= 2
            up = upsample(feed, name=str(i+1))
            concat = tf.concat([up, conv_blocks[i-1]], axis=-1, name="concat_{}".format(i))
            feed = conv_module(concat, ch, is_training, name='up_{}'.format(i), batch_norm=batch_norm,
                               pool=False)

        logits = tf.layers.conv2d(feed, num_classes, (1, 1), name='logits', activation=None, padding='same')
        return logits

