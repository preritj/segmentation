import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.misc import imresize


class ImageLoader(object):
    def __init__(self, cfg):

        self.cfg = {}
        self.cfg = cfg

    def postprocess(self, mask_pred):
        img_h, img_w = self.cfg['image_shape']
        l, r, t, b = self.cfg['crops']
        pred_mask = imresize(mask_pred, (img_h - t - b, img_w - l - r)) / 255
        real_mask = np.zeros((img_h, img_w))
        real_mask[t: img_h - b, l: img_w - r] = pred_mask
        real_mask = filter_mask(real_mask)
        return real_mask

    @staticmethod
    def load_img(img_file):
        img = cv2.imread(img_file)
        img = np.array(img, dtype=np.float32)
        return img / 127.5 - 1.

    @staticmethod
    def load_mask(mask_file):
        img_mask = plt.imread(mask_file)
        if img_mask.ndim > 2:
            img_mask = img_mask[:, :, 0] // 255
        img_mask = np.float32(img_mask)
        return np.float32(img_mask)

    def generate_rois(self, mask_files, perturb=True):
        rois = []
        for mask_file in mask_files:
            gt_bbox = get_bbox(mask_file)
            if perturb:
                pad = self.cfg['buffer']
                padded_bbox = gt_bbox + np.array([- pad, - pad, 2 * pad, 2 * pad])
                dy_c, dx_c = np.int32(0.1 * pad * (np.random.rand(2) - 0.5))
                dh, dw = np.int32(2 * pad * (np.random.rand(2) - 0.5))
                roi_bbox = (np.array(padded_bbox) +
                            np.int32([dy_c - dh/2, dx_c - dw/2, dh, dw]))
            else:
                roi_bbox = gt_bbox
            rois.append(roi_bbox)
        rois = np.array(rois)
        rois = fix_aspect_ratio(self.cfg, rois)
        return rois

    def build_roi(self, img, mask, roi, edge_factor):
        img_h, img_w = self.cfg['image_shape']
        final_roi_h, final_roi_w = self.cfg['roi_shape']
        y, x, h, w = np.array(roi, dtype=np.int32)
        y_min = max(0, y)
        y_max = min(img_h, y + h)
        x_min = max(0, x)
        x_max = min(img_w, x + w)
        roi_img = np.zeros((h, w, 3))
        roi_mask = np.zeros((h, w))
        roi_h = y_max - y_min
        roi_w = x_max - x_min
        roi_y = (h - roi_h) // 2
        roi_x = (w - roi_w) // 2
        roi_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :] = img[y_min:y_max, x_min: x_max, :]
        roi_img = cv2.resize(roi_img, (final_roi_w, final_roi_h))
        roi_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mask[y_min:y_max, x_min: x_max]
        roi_mask = cv2.resize(roi_mask, (final_roi_w, final_roi_h))
        roi_mask = np.array(np.round(roi_mask), dtype=np.uint8)

        if edge_factor > 0:
            roi_mask_weight = self.apply_edge_weighting(roi_mask, edge_factor)
        else:
            roi_mask_weight = np.ones_like(roi_mask, dtype=np.float32)

        roi_mask_weight = self.apply_class_reweighting(roi_mask, roi_mask_weight)
        return roi_img, roi_mask, roi_mask_weight

    @staticmethod
    def apply_class_reweighting(mask, mask_weight):
        n_tot = mask.size
        n_pos = np.sum(mask == 1)
        n_neg = n_tot - n_pos
        pos_wt = n_pos / n_tot
        neg_wt = n_neg / n_tot
        mask_weight[mask > 0.5] *= pos_wt
        mask_weight[mask < 0.5] *= neg_wt
        return mask_weight

    @staticmethod
    def apply_edge_weighting(mask, edge_factor):
        mask_weight = np.ones_like(mask, dtype=np.float32)
        kernel = np.ones((33, 33), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=5)
        dilation = cv2.dilate(mask, kernel, iterations=2)
        dilation[erosion > 0] = 0
        n = edge_factor
        mask_weight += n * 1.0 * dilation
        mask_weight /= (1. + n)
        return mask_weight

    def load_roi_batch(self, batch, perturb=True, edge_factor=4):
        img_files, mask_files = batch
        rois = self.generate_rois(mask_files, perturb=perturb)
        roi_imgs, roi_masks, roi_mask_weights = [], [], []
        for img_file, mask_file, roi in zip(img_files, mask_files, rois):
            img = self.load_img(img_file)
            mask = self.load_mask(mask_file)
            roi_img, roi_mask, roi_weight = self.build_roi(img, mask, roi, edge_factor)
            roi_imgs.append(roi_img)
            roi_masks.append(roi_mask)
            roi_mask_weights.append(roi_weight)
        return np.array(roi_imgs), np.array(roi_masks), np.array(roi_mask_weights)

    def preprocess_image(self, img, mask, edge_factor):
        img_h, img_w = self.cfg['image_shape']
        l, r, t, b = self.cfg['crops']
        new_img_h, new_img_w = self.cfg['scaled_img_shape']
        new_img = img[t : img_h - b, l: img_w - r, :]
        new_img = cv2.resize(new_img, (new_img_w, new_img_h), interpolation=cv2.INTER_AREA)
        if mask is None:
            return new_img
        new_mask = mask[t : img_h - b, l: img_w - r]
        new_mask = imresize(new_mask, (new_img_h, new_img_w)) / 255
        #new_mask = cv2.resize(new_mask, (new_img_w, new_img_h), interpolation=cv2.INTER_AREA)
        if edge_factor > 0:
            mask_weight = self.apply_edge_weighting(new_mask, edge_factor)
        else:
            mask_weight = np.ones_like(new_mask, dtype=np.float32)
        new_mask = np.stack((1. - new_mask, new_mask), axis=-1)
        return new_img, new_mask, mask_weight

    def load_img_batch(self, batch, edge_factor=4):
        img_files, mask_files = batch
        if mask_files is None:
            imgs = [self.load_img(f) for f in img_files]
            imgs = [self.preprocess_image(img, None, 0) for img in imgs]
            return np.array(imgs), None, None
        imgs, masks, mask_weights = [], [], []
        for img_file, mask_file in zip(img_files, mask_files):
            img = self.load_img(img_file)
            mask = self.load_mask(mask_file)
            img, mask, mask_weight = self.preprocess_image(img, mask, edge_factor)
            imgs.append(img)
            masks.append(mask)
            mask_weights.append(mask_weight)
        return np.array(imgs), np.array(masks), np.array(mask_weights)
