import os
import numpy as np
from tqdm import tqdm
from utils import get_bbox

cfg = {
    'image_shape': [1280, 1918],
    'scaled_h': 640,
    'roi_h': 640,
    'buffer': 50
}


#def extract_color_stats(img_dir, set_):
#    img_mean, img_std = [], []

def extract_bbox(mask_dir, set_):
    gt_bboxes = []
    basedir = os.path.join(mask_dir, str(set_).zfill(2))
    mask_files = [os.path.join(basedir, f) for f in os.listdir(basedir) if 'augment' not in f]
    for mask_file in tqdm(mask_files):
        gt_bboxes.append(get_bbox(mask_file))
    gt_bboxes = np.array(gt_bboxes)
    return gt_bboxes


def prepare_data_stats(args):
    """Prepare data statistics and save it in a config file"""
    set_ = args.set
    mask_dir = args.train_mask_dir
    data_stats_file = os.path.join(args.train_data_dir,
                                   str(set_).zfill(2), 'data_stats.npz')
    print("Extracting ground truth bboxes for set {}...".format(set_))
    gt_bboxes = extract_bbox(mask_dir, set_)
    if set_ != 1 and set_ != 9:
        flipped_set = 18 - set_
        print("Extracting ground truth bboxes for set {}...".format(flipped_set))
        gt_bboxes_flipped = extract_bbox(mask_dir, flipped_set)
        w = cfg['image_shape'][1]
        gt_bboxes_flipped[:,1] = w - gt_bboxes_flipped[:,1] - gt_bboxes_flipped[:,3]
        gt_bboxes = np.concatenate((gt_bboxes, gt_bboxes_flipped), axis=0)

    left_min, left_max = np.min(gt_bboxes[:, 1]), np.max(gt_bboxes[:, 1])
    top_min, top_max = np.min(gt_bboxes[:, 0]), np.max(gt_bboxes[:, 0])
    right_min = np.min(gt_bboxes[:, 1] + gt_bboxes[:, 3])
    right_max = np.max(gt_bboxes[:, 1] + gt_bboxes[:, 3])
    bottom_min = np.min(gt_bboxes[:, 0] + gt_bboxes[:, 2])
    bottom_max = np.max(gt_bboxes[:, 0] + gt_bboxes[:, 2])
    height_min, height_max = np.min(gt_bboxes[:, 2]), np.max(gt_bboxes[:, 2])
    width_min, width_max = np.min(gt_bboxes[:, 3]), np.max(gt_bboxes[:, 3])
    aspect = gt_bboxes[:, 3] / gt_bboxes[:, 2]
    aspect_min, aspect_max = np.min(aspect), np.max(aspect)
    aspect_median = np.median(aspect)
    top_std, left_std = np.std(gt_bboxes[:, :2], axis=0)
    bottom_std, right_std = np.std(gt_bboxes[:, :2] + gt_bboxes[:, 2:], axis=0)
    height_mean, width_mean = np.mean(gt_bboxes[:, 2:], axis=0)
    print("Saving data statistics to {}".format(data_stats_file))
    basedir = os.path.dirname(data_stats_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    np.savez(data_stats_file,
             left_range=np.array([left_min, left_max]),
             right_range=np.array([right_min, right_max]),
             top_range=np.array([top_min, top_max]),
             bottom_range=np.array([bottom_min, bottom_max]),
             height_range=np.array([height_min, height_max]),
             width_range=np.array([width_min, width_max]),
             aspect_stats=np.array([aspect_min, aspect_max, aspect_median]),
             mean=np.array([height_mean, width_mean]),
             std=np.array([left_std, right_std, top_std, bottom_std]))
    configure(args)


def configure(args):
    """This function determines the ideal image preprocessing such cropping and resizing
    using data statistics."""
    set_ = args.set
    data_stats_file = os.path.join(args.train_data_dir,
                                   str(set_).zfill(2), 'data_stats.npz')
    config_file = os.path.join(args.train_data_dir,
                               str(set_).zfill(2), 'config.npy')
    stats = np.load(data_stats_file)

    img_height, img_width = cfg['image_shape']
    left_crop = stats['left_range'][0] - 0.5 * stats['std'][0]
    left_crop = max(0., left_crop)
    right_crop = img_width - (stats['right_range'][1] + 0.5 * stats['std'][1])
    right_crop = max(0., right_crop)
    top_crop = stats['top_range'][0] - 0.7 * stats['std'][2]
    top_crop = max(0., top_crop)
    bottom_crop = img_height - (stats['bottom_range'][1] + 0.7 * stats['std'][3])
    bottom_crop = max(0., bottom_crop)

    if set_ == 1 or set == 9:
        left_crop = min(left_crop, right_crop)
        right_crop = left_crop

    h_cropped = img_height - top_crop - bottom_crop
    w_cropped = img_width - left_crop - right_crop
    aspect = np.round(w_cropped / h_cropped, decimals=1)
    print("Aspect ratio : ", aspect)
    delta_h = (w_cropped / aspect - h_cropped)/2
    top_crop = int(top_crop - delta_h)
    bottom_crop = int(bottom_crop - delta_h)
    left_crop = int(left_crop)
    right_crop = int(right_crop)
    crops = [left_crop, right_crop, top_crop, bottom_crop]
    print("Crop margins (l,r,t.b) : ", crops)
    cfg['crops'] = crops

    img_h = cfg['scaled_h']
    img_w = int(aspect * img_h)
    scale = img_w / w_cropped
    print("Scale factor : ", scale)
    print("Final image shape : ", [img_h, img_w])
    cfg['scale'] = scale
    cfg['scaled_img_shape'] = [img_h, img_w]

    roi_aspect = stats['aspect_stats'][2] # median
    roi_aspect = np.round(roi_aspect, decimals=1)
    roi_h = cfg['roi_h']
    roi_w = int(roi_aspect * roi_h)
    roi_shape = [roi_h, roi_w]
    print("Input roi size for segmentation : ", roi_shape)
    cfg['roi_shape'] = roi_shape
    print("Saving config data to {}".format(config_file))

    print("Extracting image mean...")

    np.save(config_file, cfg)


def load_config(train_data_dir, set_):
    cfg_file = os.path.join(train_data_dir,
                            str(set_).zfill(2), 'config.npy')
    return np.load(cfg_file).item()

    # img_h = 320
    # img_w = int(aspect * img_h)
    # if args.basic_model != 'vgg16':
    #     img_h += 1
    #     img_w += 1
    # scale = img_w/w_cropped
    # print("Scale factor : ", scale)
    # print("Final image shape : ", [img_h, img_w])
    # cfg['crops'] = crops
    # cfg['scale'] = scale
    # cfg['img_shape'] = [img_h, img_w]
    #
    # feat_stride = 16
    # print("Using feature stride of ", feat_stride)
    #
    # r_median = np.round(stats['aspect_stats'][2], decimals=1)
    # roi_h = 320
    # roi_w = int(r_median * roi_h)
    # roi_shape = [roi_h, roi_w]
    # print("Input roi size for mask-RCN : ", roi_shape)
    # cfg['roi_shape'] = roi_shape
    #
    # if args.basic_model == 'vgg16':
    #     feat_shape = [img_h//feat_stride, img_w//feat_stride, 512]
    #     roi_feat_shape = [roi_h // feat_stride, roi_w // feat_stride, 512]
    # else:
    #     feat_shape = [(img_h-1)//feat_stride+1, (img_w-1)//feat_stride+1, 2048]
    #     roi_feat_shape = [(roi_h-1)//feat_stride+1, (roi_w-1)//feat_stride+1, 2048]
    # print("Feature shape from RPN : ", feat_shape)
    # print("Feature shape from mask-RCN : ", roi_feat_shape)
    # cfg['feat_shape'] = feat_shape
    # cfg['roi_feat_shape'] = roi_feat_shape


# class ImageLoader(object):
#     def __init__(self):
#         self.img_scale = cfg['scale']
#         self.crop_margin = cfg['crops']
#
#     def load_img(self, img_file):
#         """ Load and preprocess an image. """
#         img = cv2.imread(img_file)
#
#         if self.bgr:
#             cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         left, right, top, bottom = self.crop_margin
#         img = img[top:img_height - bottom, left:img_width - right, :]
#
#         h = img_height - top - bottom
#         w = img_width - left - right
#         img = cv2.resize(img, None, fx=self.img_scale, fy=self.img_scale, interpolation=cv2.INTER_AREA)
#         img = np.float32(img)
#
#         img[:, :, 0] -= 123.68
#         img[:, :, 1] -= 116.78
#         img[:, :, 2] -= 103.94
#
#         return img
#
#     def load_imgs(self, img_files):
#         """ Load and preprocess a list of images. """
#         imgs = []
#         for img_file in img_files:
#             imgs.append(self.load_img(img_file))
#         imgs = np.array(imgs, np.float32)
#         return imgs