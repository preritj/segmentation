import time
import numpy as np
import os
from abc import abstractmethod
from tqdm import tqdm
from utils import tfrecord_util
import tensorflow as tf


class SegData(object):
    def __init__(self, cfg, img_files, mask_files):
        self.cfg = cfg
        self.img_files = img_files
        self.mask_files = mask_files

    def _create_tf_example(self, img_file, mask_file):
        feature_dict = {
            'image/filename':
                tfrecord_util.bytes_feature(img_file.encode('utf8')),
            'mask/filename':
                tfrecord_util.bytes_feature(mask_file.encode('utf8'))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tf_record(self, out_path, shuffle=True):
        print("Creating tf records : ", out_path)
        writer = tf.python_io.TFRecordWriter(out_path)
        if shuffle:
            np.random.shuffle(self.img_files)
        for img_file, mask_file in tqdm(zip(self.img_files, self.mask_files)):
            tf_example = self._create_tf_example(img_file, mask_file)
            writer.write(tf_example.SerializeToString())
        writer.close()
