import numpy as np
import os
from data_stats import prepare_data_stats
from tqdm import tqdm
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math


class DataSet(object):
    def __init__(self, args, cfg, img_files, mask_files=None):
        self.params = args
        self.is_train = (args.phase == 'train')
        self.batch_size = args.batch_size

        self.cfg = {}
        self.cfg = cfg

        self.img_files = img_files
        self.mask_files = mask_files

        self.count = len(img_files)
        self.indices = list(range(self.count))
        self.current_index = 0

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0

        if self.is_train:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = np.array(self.img_files)[current_indices]

        if (self.params.phase == 'test'):
            self.current_index += self.batch_size
            return img_files, None
        else:
            mask_files = np.array(self.mask_files)[current_indices]
            self.current_index += self.batch_size
            return img_files, mask_files


def prepare_train_data(args, cfg):
    """ Prepare data for training the model. """
    print("Preparing data for training...")
    image_dir, mask_dir, data_dir, set_ = (args.train_image_dir, args.train_mask_dir,
                                           args.train_data_dir, args.set)

    train_data_dir = os.path.join(args.train_data_dir, str(set_).zfill(2))
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    data_stats_file = os.path.join(train_data_dir, 'data_stats.npz')

    if not os.path.exists(data_stats_file):
        prepare_data_stats(args)

    img_files, mask_files = \
        prepare_data(set_, image_dir, mask_dir)

    dataset = DataSet(args, cfg, img_files, mask_files)
    return dataset


def prepare_test_data(args, cfg):
    """ Prepare data for testing the model. """
    print("Preparing data for testing...")
    image_dir, set_ = (args.test_image_dir, args.set)
    basedir = os.path.join(image_dir, str(set_).zfill(2))
    img_files = os.listdir(basedir)
    img_files = [os.path.join(basedir, f) for f in img_files]
    dataset = DataSet(args, cfg, img_files)
    return dataset


def prepare_data(set_, image_dir, mask_dir):
    img_files = os.listdir(os.path.join(image_dir, str(set_).zfill(2)))
    mask_files = []
    img_files_abs = []

    print("Building data...")
    for f in tqdm(img_files):
        tag = f.split('.jpg')[0]
        mask_file = os.path.join(mask_dir, str(set_).zfill(2), tag + '_mask')
        if "augment" in f:
            mask_file += ".png"
        else:
            mask_file += ".gif"
        mask_files.append(mask_file)
        img_files_abs.append(os.path.join(image_dir, str(set_).zfill(2), f))

    print("Dataset built.")
    return img_files_abs, mask_files


def augment(img, img_mask, data_stats_file, flip=False):
    data_stats = np.load(data_stats_file)
    left_min, left_max = data_stats['left_range']
    right_min, right_max = data_stats['right_range']
    top_min, top_max = data_stats['top_range']
    bottom_min, bottom_max = data_stats['bottom_range']
    height_min, height_max = data_stats['height_range']
    width_min, width_max = data_stats['width_range']

    l = random.randint(left_min, left_max)
    t = random.randint(top_min, top_max)
    max_h = min(height_max, bottom_max - t)
    max_w = min(width_max, right_max - l)
    min_h = max(height_min, bottom_min - t)
    min_w = max(width_min, right_min - l)
    t0, l0 = np.min(np.nonzero(img_mask), axis=1)
    b0, r0 = np.max(np.nonzero(img_mask), axis=1)
    h0, w0 = (b0 - t0), (r0 - l0)
    rw_min = min_w/w0
    rw_max = max_w/w0
    rh_min = min_h/h0
    rh_max = max_h/h0
    r_min = max(rw_min, rh_min)
    r_max = min(rw_max, rh_max)
    ratio = random.uniform(r_min, r_max)
    r = l + w0 * ratio
    b = t + h0 * ratio
    pts1 = np.float32([[l0, t0], [r0, t0], [r0, b0]])
    pts2 = np.float32([[l, t], [r, t], [r, b]])
    mat = cv2.getAffineTransform(pts1, pts2)
    if r < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    rows, cols, _ = img.shape
    new_img = cv2.warpAffine(img, mat, (cols, rows), flags=interpolation)
    new_img_mask = cv2.warpAffine(img_mask, mat, (cols, rows), flags=interpolation)

    rot_angle = random.uniform(-1, 1)
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1)
    new_img = cv2.warpAffine(new_img, mat, (cols, rows))
    new_img_mask = cv2.warpAffine(new_img_mask, mat, (cols, rows))
    if flip:
        if random.randint(0, 1):
            new_img = cv2.flip(new_img, 1)
            new_img_mask = cv2.flip(new_img_mask, 1)
    hsv = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    hsv = np.float32(hsv)
    hue_shift = random.randint(-50, 50)
    hsv[:, :, 0][new_img_mask == 1] += hue_shift
    hsv[:, :, 0][hsv[:, :, 0] < 0] += 180
    val_scale = random.uniform(0.75, 1.25)
    hsv[:, :, 2] *= val_scale
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    val_scale = random.uniform(0.75, 1.25)
    hsv[:, :, 1] *= val_scale
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv = np.uint8(hsv)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img, new_img_mask


def augment_data(args):
    set_ = args.set
    image_dir = args.train_image_dir
    mask_dir = args.train_mask_dir
    data_stats_file = os.path.join(args.train_data_dir,
                                   str(set_).zfill(2), 'data_stats.npz')
    num_aug = args.augment_factor
    img_files = os.listdir(os.path.join(image_dir, str(set_).zfill(2)))
    print("Removing old augmentations...")
    for f in img_files:
        if "augment" in f:
            os.remove(os.path.join(image_dir, str(set_).zfill(2), f))

    flip_data = True
    num_aug_flipped = 0
    print("Creating augmented dataset...")

    if (set_ != 1) and (set_ != 9):
        num_aug_flipped = int(math.ceil(num_aug / 2))
        flipped_set = 18 - set_
        flip_data = False
        flipped_img_files = os.listdir(os.path.join(image_dir, str(flipped_set).zfill(2)))
        for f in tqdm(flipped_img_files):
            tag = f.split('.jpg')[0]
            s = flipped_set
            img_file = os.path.join(image_dir, str(s).zfill(2), tag + '.jpg')
            mask_file = os.path.join(mask_dir, str(s).zfill(2), tag + '_mask.gif')
            img = plt.imread(img_file)
            img_mask = plt.imread(mask_file)[:, :, 0] // 255
            img = cv2.flip(img, 1)
            img_mask = cv2.flip(img_mask, 1)

            for n_aug in range(num_aug_flipped):
                new_img, new_img_mask = augment(img, img_mask, data_stats_file, flip_data)
                new_img_file = os.path.join(image_dir, str(set_).zfill(2), tag +
                                            '_augment' + str(n_aug).zfill(2) + '.jpg')
                new_mask_file = os.path.join(mask_dir, str(set_).zfill(2), tag +
                                             '_augment' + str(n_aug).zfill(2) + '_mask.png')
                new_img = Image.fromarray(new_img)
                new_img.save(new_img_file)
                cv2.imwrite(new_mask_file, 255 * new_img_mask.astype(np.uint8))

    img_files = os.listdir(os.path.join(image_dir, str(set_).zfill(2)))
    for f in tqdm(img_files):
        tag = f.split('.jpg')[0]
        s = set_
        img_file = os.path.join(image_dir, str(s).zfill(2), tag + '.jpg')
        mask_file = os.path.join(mask_dir, str(s).zfill(2), tag + '_mask.gif')
        img = plt.imread(img_file)
        img_mask = plt.imread(mask_file)[:, :, 0] // 255

        for n_aug in range(num_aug_flipped, num_aug):
            new_img, new_img_mask = augment(img, img_mask, data_stats_file, flip_data)
            new_img_file = os.path.join(image_dir, str(set_).zfill(2), tag +
                                        '_augment' + str(n_aug).zfill(2) + '.jpg')
            new_mask_file = os.path.join(mask_dir, str(set_).zfill(2), tag +
                                         '_augment' + str(n_aug).zfill(2) + '_mask.png')
            new_img = Image.fromarray(new_img)
            new_img.save(new_img_file)
            cv2.imwrite(new_mask_file, 255 * new_img_mask.astype(np.uint8))
