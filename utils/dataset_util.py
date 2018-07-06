import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def random_int(maxval, minval=0):
    return tf.random_uniform(
        shape=[], minval=minval, maxval=maxval, dtype=tf.int32)


def rotate(image, k):
    # k = np.random.randint(0, 4)
    if k > 0:
        image = tf.image.rot90(image, k)
    return image


def flip_left_right(img):
    # random_var = random_int(2)
    # random_var = tf.cast(random_var, tf.bool)
    # flipped_img = tf.cond(random_var,
    #                       true_fn=lambda: tf.image.flip_left_right(img),
    #                       false_fn=lambda: tf.identity(img))
    # mask = tf.expand_dims(mask, axis=2)
    # flipped_mask = tf.cond(random_var,
    #                        true_fn=lambda: tf.image.flip_left_right(mask),
    #                        false_fn=lambda: tf.identity(mask))
    # flipped_mask = tf.squeeze(flipped_mask)
    # if weights is None:
    #     return flipped_img, flipped_mask
    # weights = tf.expand_dims(mask, axis=2)
    # flipped_weights = tf.cond(
    #     random_var,
    #     true_fn=lambda: tf.image.flip_left_right(weights),
    #     false_fn=lambda: tf.identity(weights))
    # flipped_weights = tf.squeeze(flipped_weights)
    flipped_image = tf.image.flip_left_right(img)
    return flipped_image


def random_brightness(image):
    image = tf.image.random_brightness(
        image,
        max_delta=0.1)
    return image


def random_contrast(image):
    image = tf.image.random_contrast(
        image,
        lower=0.9,
        upper=1.1)
    return image


def random_hue(image):
    image = tf.image.random_hue(
        image,
        max_delta=0.1)
    return image


# def resize(image, keypoints, bbox, mask,
#            target_image_size=(224, 224),
#            target_mask_size=None):
#     img_size = list(target_image_size)
#     if target_mask_size is None:
#         target_mask_size = img_size
#     mask_size = list(target_mask_size)
#     new_image = tf.image.resize_images(image, size=img_size)
#     new_mask = tf.expand_dims(mask, axis=2)
#     new_mask.set_shape([None, None, 1])
#     new_mask = tf.image.resize_images(new_mask, size=mask_size)
#     new_mask = tf.squeeze(new_mask)
#     return new_image, keypoints, bbox, new_mask

###################################################
# Some other potentially useful functions

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def dice_coef(y_true, y_pred, smooth=0.001):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return np.mean(dice)


def get_bbox(mask_file):
    binary_img = plt.imread(mask_file)
    if binary_img.ndim > 2:
        binary_img = binary_img[:, :, 0] // 255
    ymin, xmin = np.min(np.nonzero(binary_img), axis=1)
    ymax, xmax = np.max(np.nonzero(binary_img), axis=1)
    return [ymin - 1, xmin - 1, ymax - ymin + 2, xmax - xmin + 2]


def fix_aspect_ratio(cfg, rois):
    h_roi, w_roi = cfg['roi_shape']
    roi_aspect = w_roi / h_roi

    aspect_rois = rois[:, 3] / rois[:, 2]
    idx_a = np.where(aspect_rois > roi_aspect)[0]
    idx_b = np.where(aspect_rois < roi_aspect)[0]

    rois_a = rois[idx_a, :]
    desired_h = rois_a[:, 3] / roi_aspect
    delta_h = (desired_h - rois_a[:, 2]) / 2
    rois_a[:, 0] = rois_a[:, 0] - delta_h
    rois_a[:, 2] = desired_h

    rois_b = rois[idx_b, :]
    desired_w = rois_b[:, 2] * roi_aspect
    delta_w = (desired_w - rois_b[:, 3]) / 2
    rois_b[:, 1] = rois_b[:, 1] - delta_w
    rois_b[:, 3] = desired_w

    rois[idx_a, :] = rois_a
    rois[idx_b, :] = rois_b

    return rois


def filter_mask(mask):
    kernel = np.ones((2, 2))
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_smooth = np.uint8(np.round(mask_smooth))
    blobs = cv2.connectedComponentsWithStats(mask_smooth, 4, cv2.CV_32S)
    stats = blobs[2]
    obj_label = None
    for i, stat in enumerate(stats):
        if stat[4] < 10000:
            continue
        elif (stat[0] < 2) and (stat[1] < 2):
            continue
        else:
            obj_label = i
            break
    blobs = blobs[1]
    blobs[blobs != obj_label] = 0
    return np.uint8(blobs)




