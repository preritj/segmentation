from models.losses import pixel_wise_loss
from abc import abstractmethod


class SegModel(object):
    def __init__(self, model_cfg):
        self.cfg = model_cfg

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def build_net(self, preprocessed_inputs, is_training=False):
        """Builds network and returns heatmaps and fpn features"""
        raise NotImplementedError("Not yet implemented")

    def predict(self, inputs, is_training=False):
        images = inputs['images']
        preprocessed_inputs = self.preprocess(images)
        mask_logits = self.build_net(
            preprocessed_inputs, is_training=is_training)
        prediction = {'mask_logits': mask_logits}
        return prediction

    def losses(self, prediction, ground_truth):
        mask_logits = prediction['mask_logits']
        masks_gt = ground_truth['masks']
        weights_gt = None
        if self.cfg.use_weights:
            weights_gt = ground_truth['weights']
        loss = pixel_wise_loss(mask_logits, masks_gt, pixel_weights=weights_gt)
        losses = {'CE_loss': loss}
        return losses


    # def create_tf_placeholders(self):
    #     roi_h, roi_w = self.cfg['scaled_img_shape'] #self.cfg['roi_shape']
    #     roi_images = tf.placeholder(tf.float32, [self.batch_size, roi_h, roi_w, 3])
    #     roi_masks = tf.placeholder(tf.float32, [self.batch_size, roi_h, roi_w, 2])
    #     roi_weights = tf.placeholder(tf.float32, [self.batch_size, roi_h, roi_w])
    #     self.tf_placeholders = {'images': roi_images,
    #                             'masks': roi_masks,
    #                             'weights': roi_weights}


    # def make_train_op(self):
    #     learning_rate = self.params.learning_rate
    #     roi_masks = self.tf_placeholders["masks"]
    #     roi_masks_pos = tf.slice(roi_masks, [0, 0, 0, 1], [-1, -1, -1, 1])
    #     roi_masks_pos = tf.squeeze(roi_masks_pos, [-1])
    #     roi_weights = self.tf_placeholders["weights"]
    #     _, tf_mask = mask_prediction(self.mask_logits)
    #     loss0 = dice_coef_loss(roi_masks_pos, tf_mask)
    #     loss1 = pixel_wise_loss(self.mask_logits, roi_masks, pixel_weights=roi_weights)
    #     loss = loss0 + self.params.sce_weight * loss1
    #     solver = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    #
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         self.train_op = solver.minimize(loss, global_step=self.global_step)
    #         self.loss_op = [loss0, loss1]
    #
    # def make_eval_op(self):
    #     pred_probs, pred_masks = mask_prediction(self.mask_logits)
    #     self.eval_op = [pred_probs, pred_masks]
    #
    # def get_feed_dict(self, batch, perturb=True):
    #     if self.stage == 1:
    #         roi_images, roi_masks, roi_weights = \
    #         self.image_loader.load_img_batch(batch, edge_factor=self.params.edge_factor)
    #     else:
    #         roi_images, roi_masks, roi_weights = \
    #             self.image_loader.load_roi_batch(batch, perturb=perturb,
    #                                          edge_factor=self.params.edge_factor)
    #     tf_roi_images = self.tf_placeholders["images"]
    #     if roi_masks is None:
    #         return {tf_roi_images: roi_images}
    #     tf_roi_masks = self.tf_placeholders["masks"]
    #     tf_roi_weights = self.tf_placeholders["weights"]
    #     return {tf_roi_images: roi_images,
    #             tf_roi_masks: roi_masks,
    #             tf_roi_weights: roi_weights}
    #
    # def train(self, data):
    #     """ Train the model. """
    #     params = self.params
    #     save_dir = os.path.join(params.save_dir, str(params.set).zfill(2), 'stage_'+str(self.stage))
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     save_dir = os.path.join(save_dir, 'model')
    #     self.make_train_op()
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #         if params.load:
    #             self.load(sess, saver)
    #
    #         n_display = params.display_period
    #         for i_epoch in tqdm(list(range(params.num_epochs)), desc='epoch'):
    #             dice_loss, sce_loss, n_steps = 0, 0, 0
    #             for _ in tqdm(list(range(0, data.count, self.batch_size)), desc='batch'):
    #                 batch = data.next_batch()
    #                 if len(batch[0]) < self.batch_size:
    #                     continue
    #                 ops = [self.train_op, self.global_step] + self.loss_op
    #                 feed_dict = self.get_feed_dict(batch, perturb=True)
    #                 _, global_step, loss0, loss1 = sess.run(ops, feed_dict=feed_dict)
    #                 if n_steps + 1 == n_display:
    #                     print("Dice coeff : {}, Cross entropy loss : {}"
    #                           .format(-dice_loss/n_steps, sce_loss/n_steps))
    #                     dice_loss, sce_loss, n_steps = 0, 0, 0
    #                 else:
    #                     dice_loss += loss0
    #                     sce_loss += loss1
    #                     n_steps += 1
    #
    #                 if (global_step + 1) % params.save_period == 0:
    #                     print("Saving model in {}".format(save_dir))
    #                     saver.save(sess, save_dir, global_step)
    #             data.reset()
    #             print("{} epochs finished.".format(i_epoch))
    #
    # def validate(self, data):
    #     """ Test the model. """
    #     # params = self.params
    #     self.make_eval_op()
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #         self.load(sess, saver)
    #         for _ in tqdm(list(range(data.count)), desc='batch'):
    #             batch = data.next_batch()
    #             img_file, mask_file = batch[0][0], batch[1][0]
    #
    #             gt_bbox = self.image_loader.generate_rois([mask_file], perturb=False)[0]
    #             feed_dict = self.get_feed_dict(batch, perturb=False)
    #             pred_probs, _ = sess.run(self.eval_op, feed_dict=feed_dict)
    #             # pred_mask = np.zeros_like(pred_probs, dtype=np.uint8)
    #             # pred_mask[np.where(pred_mask > 0.5)] = 1
    #             # print(np.where(pred_mask > 0.5))
    #             mask_pred = pred_probs[0, :, :, 1]
    #             mask_pred[mask_pred > 0.5] = 1
    #             mask_pred[mask_pred <= 0.5] = 0
    #
    #             if True:
    #                 img = cv2.imread(img_file)
    #                 real_mask = np.zeros_like(img, dtype=np.uint8)
    #                 if self.stage == 1:
    #                     img_h, img_w = self.cfg['image_shape']
    #                     l, r, t, b = self.cfg['crops']
    #                     pred_mask = imresize(mask_pred, (img_h - t - b, img_w - l - r)) / 255
    #                     real_mask[t: img_h - b, l: img_w - r, 0] = np.uint8(np.round(pred_mask))
    #                 else:
    #                     y, x, h, w = gt_bbox
    #                     pred_mask = cv2.resize(mask_pred, (w, h))
    #                     real_mask[y:y + h, x:x + w, 0] = np.uint8(pred_mask)
    #
    #
    #                 winname = 'Image %s' % (img_file)
    #                 img = cv2.resize(img, (1438, 960))
    #                 img_mask = cv2.resize(real_mask * 255, (1438, 960), interpolation=cv2.INTER_CUBIC)
    #                 display_img = cv2.addWeighted(img, 0.2, img_mask, 0.8, 0)
    #                 cv2.imshow(winname, display_img)
    #                 cv2.moveWindow(winname, 100, 100)
    #                 cv2.waitKey(1000)
    #
    #                 gt_mask = self.image_loader.load_mask(mask_file)
    #                 print("Dice coefficient : ", dice_coef(gt_mask, real_mask[:,:,0]))
    #
    # def test(self, data):
    #     """ Test the model. """
    #     params = self.params
    #     self.make_eval_op()
    #
    #     res_dir = params.test_results_dir
    #     res_dir = os.path.join(res_dir, str(params.set).zfill(2),
    #                            'stage_' + str(params.stage))
    #     if not os.path.exists(res_dir):
    #         os.makedirs(res_dir)
    #     img_names = []
    #     rle_strings = []
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #         self.load(sess, saver)
    #         for _ in tqdm(list(range(data.count)), desc='batch'):
    #             batch = data.next_batch()
    #             img_file = batch[0][0]
    #             feed_dict = self.get_feed_dict(batch, perturb=False)
    #             pred_probs, _ = sess.run(self.eval_op, feed_dict=feed_dict)
    #             # pred_mask = np.zeros_like(pred_probs, dtype=np.uint8)
    #             # pred_mask[np.where(pred_mask > 0.5)] = 1
    #             # print(np.where(pred_mask > 0.5))
    #             mask_pred = pred_probs[0, :, :]
    #             #mask_pred[mask_pred > 0.5] = 1
    #             #mask_pred[mask_pred <= 0.5] = 0
    #             real_mask = self.image_loader.postprocess(mask_pred)
    #             rle = rle_encode(real_mask)
    #             rle_strings.append(rle_to_string(rle))
    #
    #             if 1:
    #                 img = cv2.imread(img_file)
    #                 img_mask = np.zeros_like(img)
    #                 img_mask[:, :, 0] = real_mask * 255
    #                 # y, x, h, w = gt_bbox
    #                 # print(gt_bbox)
    #
    #                 winname = 'Image %s' % (img_file)
    #                 img = cv2.resize(img, (1438, 960))
    #
    #
    #                 img_mask = cv2.resize(img_mask, (1438, 960))
    #                 display_img = cv2.addWeighted(img, 0.4, img_mask, 0.6, 0)
    #                 cv2.imshow(winname, display_img)
    #                 cv2.moveWindow(winname, 100, 100)
    #                 cv2.waitKey(1000)
    #
    #             img_name = os.path.basename(img_file)
    #             img_names.append(img_name)
    #             #outfile = os.path.join(res_dir, str(img_name) + '.npy')
    #             #np.save(outfile, mask_pred)
    #         df = {'img' : img_names, 'rle_mask' : rle_strings}
    #         df = pd.DataFrame(df)
    #         outfile = os.path.join(res_dir, 'results.csv')
    #         df.to_csv(outfile)
    #
    #
    #
    # def load(self, sess, saver):
    #     """ Load the trained model. """
    #     params = self.params
    #     print("Loading model...")
    #     load_dir = os.path.join(params.save_dir, str(params.set).zfill(2),
    #                             'stage_'+str(params.stage), 'model')
    #     checkpoint = tf.train.get_checkpoint_state(os.path.dirname(load_dir))
    #     if checkpoint is None:
    #         print("Error: No saved model found. Please train first.")
    #         sys.exit(0)
    #     saver.restore(sess, checkpoint.model_checkpoint_path)