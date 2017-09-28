#!/usr/bin/env python

import argparse
import sys
from model import Model
from data_stats import prepare_data_stats, load_config
from data import prepare_train_data, prepare_test_data, augment_data


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='fcn_gcn', help='NN to use: can be unet or fcn_gcn')
    parser.add_argument('--phase', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Turn on to load the pretrained model')

    parser.add_argument('--prepare_data_stats', action='store_true', default=False,
                        help='Turn on to prepare data statistics. Must do this for the first time of training.')

    parser.add_argument('--set', type=int, default=1,
                        help='set for one of the zones/angles: Can be integer from 1 to 16')

    parser.add_argument('--train_image_dir', default='../data/train/images/',
                        help='Directory containing training images')
    parser.add_argument('--train_mask_dir', default='../data/train/masks/',
                        help='Directory containing masks for training images')
    parser.add_argument('--train_data_dir', default='../data/train/misc/',
                        help='Directory to store temporary training data')
    parser.add_argument('--test_image_dir', default='../data/test/images/',
                        help='Directory containing test images')
    parser.add_argument('--test_results_dir', default='../data/test/results/',
                        help='Directory containing results for test set')

    parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model')

    parser.add_argument('--save_period', type=int, default=100, help='Period to save the trained model')
    parser.add_argument('--display_period', type=int, default=20,
                        help='Period over which to display loss.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                        help='Turn on to use batch normalization')
    parser.add_argument('--sce_weight', type=float, default=1.,
                        help='Adds softmax cross-entropy (SCE) loss when weight is non-zero')
    parser.add_argument('--edge_factor', type=int, default=0,
                        help='Gives additional weight to edges when using SCE')

    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Turn on to generate augmented data for the first time')
    parser.add_argument('--augment_factor', type=int, default=1,
                        help='Factor by which to augment original data')

    args = parser.parse_args()

    if args.prepare_data_stats:
        prepare_data_stats(args)

    if args.augment_data:
        augment_data(args)

    cfg = load_config(args.train_data_dir, args.set)
    model = Model(args, cfg)

    if args.phase == 'train':
        train_data = prepare_train_data(args, cfg)
        model.train(train_data)
    elif args.phase == 'val':
        assert args.batch_size == 1
        train_data = prepare_train_data(args, cfg)
        model.validate(train_data)
    elif args.phase == 'test':
        assert args.batch_size == 1
        test_data = prepare_test_data(args, cfg)
        model.test(test_data)
    else:
        return

if __name__ == "__main__":
    main(sys.argv)

