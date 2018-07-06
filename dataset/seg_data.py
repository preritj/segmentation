import time
import numpy as np
import os
from abc import abstractmethod
from tqdm import tqdm
from utils import tfrecord_util
import tensorflow as tf


class SegData(object):
    def __init__(self, cfg, img_files, mask_files=None):
        """
        Constructor of ObjectData class
        """
        self.cfg = cfg
        self.imgs, self.ids = None, None
        self.data_dir = data_dir
        self.product_labels = {}
        print('loading annotations into memory...')
        tic = time.time()
        self.datasets = []
        if type(train_files) != list:
            train_files = [train_files]
        for train_file in train_files:
            labels_file = os.path.dirname(train_file)
            labels_file = os.path.join(labels_file, 'labels.txt')
            with open(labels_file, 'r') as f:
                self.product_names = {}
                for line in f:
                    label, prod_name = line.split()
                    self.product_labels[prod_name] = int(label)
            with open(train_file, 'r') as f:
                dataset = {}
                for line in f:
                    img, ann_file = line.split()
                    dataset[img] = ann_file
                self.datasets.append(dataset)
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        self.create_index()

    @abstractmethod
    def create_index(self):
        return

    def get_size(self):
        return len(self.ids)

    def _create_tf_example(self, img_id):
        img_meta = self.imgs[img_id]
        img_file = img_meta['filename']
        img_file = os.path.join(self.data_dir, img_file)
        img_shape = list(img_meta['shape'])

        feature_dict = {
            'image/filename':
                tfrecord_util.bytes_feature(img_file.encode('utf8')),
            'image/shape':
                tfrecord_util.int64_list_feature(img_shape),
            'mask/filename':
                tfrecord_util.bytes_feature(img_file.encode('utf8'))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tf_record(self, out_path, shuffle=True):
        print("Creating tf records : ", out_path)
        writer = tf.python_io.TFRecordWriter(out_path)
        if shuffle:
            np.random.shuffle(self.ids)
        for img_id in tqdm(self.ids):
            tf_example = self._create_tf_example(img_id)
            writer.write(tf_example.SerializeToString())
        writer.close()
