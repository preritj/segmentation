import os
import glob
import numpy as np
import tensorflow as tf
import functools
from utils.dataset_util import (
    rotate, random_hue, random_contrast, random_brightness)
from dataset.seg_data import SegData


slim_example_decoder = tf.contrib.slim.tfexample_decoder


class SegDataReader(object):
    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        self.datasets = []

        for dataset in data_cfg.datasets:
            data_dir = dataset['data_dir']
            name = dataset['name']
            weight = dataset['weight']
            img_files = os.path.join(
                data_dir, 'images', '*')
            mask_files = [os.path.join(
                data_dir, 'masks', os.path.basename(f).split('.')[0] + '.png')
                for f in img_files]
            tfrecord_file = dataset['tfrecord_file']
            if (tfrecord_file is None) or dataset['overwrite_tfrecord']:
                tfrecord_name = os.path.basename(data_dir) + '.records'
                sub_dir = os.path.dirname(dataset['tfrecord_files'])
                tfrecord_path = os.path.join(data_dir, sub_dir, tfrecord_name)
                tfrecord_dir = os.path.dirname(tfrecord_path)
                if not os.path.exists(tfrecord_dir):
                    os.makedirs(tfrecord_dir)
                ds = self.add_dataset(name, img_files, mask_files)
                ds.create_tf_record(tfrecord_path)
                self.datasets.append({'name': name,
                                      'tfrecord_path': tfrecord_path,
                                      'weight': weight})

    def add_dataset(self, name, img_files, mask_files):
        if name == 'png_objects':
            ds = SegData(self.data_cfg, img_files, mask_files)
        else:
            raise RuntimeError('Dataset not supported')
        return ds

    def _get_probs(self):
        probs = [ds['weight'] for ds in self.datasets]
        probs = np.array(probs)
        return probs / np.sum(probs)

    @staticmethod
    def _get_tensor(tensor):
        if isinstance(tensor, tf.SparseTensor):
            return tf.sparse_tensor_to_dense(tensor)
        return tensor

    @staticmethod
    def _image_decoder(keys_to_tensors):
        filename = keys_to_tensors['image/filename']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded

    @staticmethod
    def _mask_decoder(keys_to_tensors):
        filename = keys_to_tensors['mask/filename']
        mask_string = tf.read_file(filename)
        mask_decoded = tf.image.decode_png(mask_string)
        return mask_decoded

    def _decoder(self):
        keys_to_features = {
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'mask/filename':
                tf.FixedLenFeature((), tf.string, default_value='')
        }
        items_to_handlers = {
            'image': slim_example_decoder.ItemHandlerCallback(
                'image/filename', self._image_decoder),
            'mask': slim_example_decoder.ItemHandlerCallback(
                'mask/filename', self._mask_decoder)
        }
        decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        return decoder

    def augment_data(self, dataset, train_cfg):
        aug_cfg = train_cfg.augmentation
        preprocess_cfg = train_cfg.preprocess
        img_size = preprocess_cfg['image_resize']
        if aug_cfg['flip_left_right']:
            random_flip_left_right_fn = functools.partial(
                random_flip_left_right,
                flipped_keypoint_indices=flipped_kp_indices)
            dataset = dataset.map(
                random_flip_left_right_fn,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        random_crop_fn = functools.partial(
            random_crop,
            crop_size=img_size,
            scale_range=aug_cfg['scale_range']
        )
        if aug_cfg['random_crop']:
            dataset = dataset.map(
                random_crop_fn,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_brightness']:
            dataset = dataset.map(
                random_brightness,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_contrast']:
            dataset = dataset.map(
                random_contrast,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def preprocess_data(self, dataset, train_cfg):
        preprocess_cfg = train_cfg.preprocess
        img_size = preprocess_cfg['image_resize']
        resize_fn = functools.partial(
            resize,
            target_image_size=img_size)
        dataset = dataset.map(
            resize_fn,
            num_parallel_calls=train_cfg.num_parallel_map_calls
        )
        dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def read_data(self, train_config):
        probs = self._get_probs()
        probs = tf.cast(probs, tf.float32)
        decoder = self._decoder()
        filenames = [ds['tfrecord_path'] for ds in self.datasets]
        file_ids = list(range(len(filenames)))
        dataset = tf.data.Dataset.from_tensor_slices((file_ids, filenames))
        dataset = dataset.apply(tf.contrib.data.rejection_resample(
            class_func=lambda c, _: c,
            target_dist=probs,
            seed=42))
        dataset = dataset.map(lambda _, a: a[1])
        if train_config.shuffle:
            dataset = dataset.shuffle(
                train_config.filenames_shuffle_buffer_size)

        dataset = dataset.repeat(train_config.num_epochs or None)

        file_read_func = functools.partial(tf.data.TFRecordDataset,
                                           buffer_size=8 * 1000 * 1000)
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                file_read_func, cycle_length=train_config.num_readers,
                block_length=train_config.read_block_length, sloppy=True))
        if train_config.shuffle:
            dataset = dataset.shuffle(train_config.shuffle_buffer_size)

        decode_fn = functools.partial(
            decoder.decode, items=['image', 'mask'])
        dataset = dataset.map(
            decode_fn, num_parallel_calls=train_config.num_parallel_map_calls)
        dataset = dataset.prefetch(train_config.prefetch_size)

        dataset = self.augment_data(dataset, train_config)

        dataset = self.preprocess_data(dataset, train_config)
        return dataset
