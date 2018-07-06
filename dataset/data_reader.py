import os
import glob
import numpy as np
import tensorflow as tf
import functools
from utils.dataset_util import (
    rotate, random_hue, random_contrast, random_brightness)


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
                data_dir, dataset['img_glob_string'])
            mask_files = dataset['img_glob_string']
            if mask_files is not None:
                img_files = os.path.join(
                    data_dir, dataset['img_glob_string'])
            tfrecord_file = dataset['tfrecord_file']
            if (tfrecord_file is None) or dataset['overwrite_tfrecord']:
                tfrecord_name = os.path.basename(data_dir) + '.records'
                sub_dir = os.path.dirname(dataset['tfrecord_files'])
                tfrecord_path = os.path.join(data_dir, sub_dir, tfrecord_name)
                tfrecord_dir = os.path.dirname(tfrecord_path)
                if not os.path.exists(tfrecord_dir):
                    os.makedirs(tfrecord_dir)
                ds = self.add_dataset(name, img_files)
                ds.create_tf_record(tfrecord_path)
                self.datasets.append({'name': name,
                                      'tfrecord_path': tfrecord_path,
                                      'weight': weight})

    def add_dataset(self, name, img_files, mask_files):
        if name == 'png_objects':
            ds = PNGobjects(self.data_cfg, img_files, mask_files)
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
        # TODO: decode after crop to increase speed
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        return image_decoded

    @staticmethod
    def _mask_decoder(keys_to_tensors):
        mask_x = SegDataReader._get_tensor(
            keys_to_tensors['image/mask/x'])
        mask_y = SegDataReader._get_tensor(
            keys_to_tensors['image/mask/y'])
        shape = SegDataReader._get_tensor(
            keys_to_tensors['image/shape'])

        indices = tf.stack([mask_x, mask_y], axis=1)
        values = tf.zeros_like(mask_x)
        mask = tf.SparseTensor(indices=indices, values=values,
                               dense_shape=shape)
        # TODO: possibly do sparse to dense coversion after crop
        mask = tf.sparse_tensor_to_dense(mask, default_value=1)
        return tf.cast(mask, tf.int32)

    @staticmethod
    def _keypoints_decoder(keys_to_tensor, num_keypoints=15):
        keypoints = keys_to_tensor['image/person/keypoints']
        img_shape = keys_to_tensor['image/shape']
        num_instances = PoseDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances] + [num_keypoints, 3]
        if isinstance(keypoints, tf.SparseTensor):
            keypoints = tf.sparse_reshape(keypoints, shape=shape)
            keypoints = tf.sparse_tensor_to_dense(keypoints)
        else:
            keypoints = tf.reshape(keypoints, shape=shape)
        keypoints = normalize_keypoints(keypoints, img_shape)
        return keypoints

    @staticmethod
    def _bbox_decoder(keys_to_tensor):
        bbox = keys_to_tensor['image/person/bbox']
        img_shape = keys_to_tensor['image/shape']
        num_instances = PoseDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances] + [4]
        if isinstance(bbox, tf.SparseTensor):
            bbox = tf.sparse_reshape(bbox, shape=shape)
            bbox = tf.sparse_tensor_to_dense(bbox)
        else:
            bbox = tf.reshape(bbox, shape=shape)
        bbox = normalize_bboxes(bbox, img_shape)
        return bbox

    def _decoder(self):
        keys_to_features = {
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/shape':
                tf.FixedLenFeature([2], tf.int64),
            'image/num_instances':
                tf.FixedLenFeature((), tf.int64),
            'image/person/bbox':
                tf.VarLenFeature(tf.float32),
            'image/person/keypoints':
                tf.VarLenFeature(tf.float32),
            'image/mask/x':
                tf.VarLenFeature(tf.int64),
            'image/mask/y':
                tf.VarLenFeature(tf.int64)
        }
        keypoints_decoder = functools.partial(
            self._keypoints_decoder, num_keypoints=self.num_keypoints)
        items_to_handlers = {
            'image': slim_example_decoder.ItemHandlerCallback(
                'image/filename', self._image_decoder),
            'mask': slim_example_decoder.ItemHandlerCallback(
                ['image/mask/x', 'image/mask/y', 'image/shape'],
                self._mask_decoder),
            'keypoints': slim_example_decoder.ItemHandlerCallback(
                ['image/person/keypoints', 'image/num_instances',
                 'image/shape'], keypoints_decoder),
            'bbox': slim_example_decoder.ItemHandlerCallback(
                ['image/person/bbox', 'image/num_instances',
                 'image/shape'], self._bbox_decoder)
        }
        decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        return decoder

    def augment_data(self, dataset, train_cfg):
        aug_cfg = train_cfg.augmentation
        preprocess_cfg = train_cfg.preprocess
        img_size = preprocess_cfg['image_resize']
        if aug_cfg['flip_left_right']:
            kp_dict = {kp_name: i for i, kp_name in
                       enumerate(train_cfg.train_keypoints)}
            flipped_kp_indices = []
            for kp_name in train_cfg.train_keypoints:
                if kp_name.startswith('left'):
                    flipped_kp_name = 'right' + kp_name.split('left')[1]
                    flipped_kp_indices.append(kp_dict[flipped_kp_name])
                elif kp_name.startswith('right'):
                    flipped_kp_name = 'left' + kp_name.split('right')[1]
                    flipped_kp_indices.append(kp_dict[flipped_kp_name])
                else:
                    flipped_kp_indices.append(kp_dict[kp_name])
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
            decoder.decode, items=['image', 'keypoints', 'bbox', 'mask'])
        dataset = dataset.map(
            decode_fn, num_parallel_calls=train_config.num_parallel_map_calls)
        dataset = dataset.prefetch(train_config.prefetch_size)

        train_keypoints = [self.data_cfg.keypoints[kp_name]
                           for kp_name in train_config.train_keypoints]
        kp_subset_fn = functools.partial(
            keypoints_select, keypoints_to_keep=train_keypoints)
        dataset = dataset.map(
            kp_subset_fn,
            num_parallel_calls=train_config.num_parallel_map_calls)
        dataset = dataset.prefetch(train_config.prefetch_size)

        dataset = self.augment_data(dataset, train_config)

        dataset = self.preprocess_data(dataset, train_config)
        return dataset
