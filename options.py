import os
import random
import argparse
import numpy as np
import tensorflow as tf


def str2bool(v):
    """
    Transform string inputs into boolean.
    :param v: string input.
    :return: string input transformed to boolean.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    """Set seed"""
    if seed is None:
        seed = random.randrange(100)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    tf.random.set_seed(seed)  # Tensorflow module


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="InPercept: Vehicle detection with adverse conditions (night, rain, fog...)")

    # Seed
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility. None to use random seed')

    # Input
    parser.add_argument('--img_width', type=int, default=224, help="Target Image Width")
    parser.add_argument('--img_height', type=int, default=224, help="Target Image Height")
    parser.add_argument('--img_mode', type=str, default='rgb', help="Load mode for images, either rgb or grayscale")
    parser.add_argument('--h_grid', type=int, default=28, help="Horizontal size of the grid classifiers")
    parser.add_argument('--v_grid', type=int, default=28, help="Vertical size of the grid classifiers")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--initial_epoch', type=int, default=0, help="Initial epoch to start training")
    parser.add_argument('--initial_lr', type=int, default=1e-3, help="Initial learning rate for Adam")
    parser.add_argument('--data_aug', type=str2bool, default=False, help="Apply data augmentation")

    # Testing parameters
    parser.add_argument('--d_th', type=float, default=16, help="Distances between markers higher than this threshold"
                                                               "are filtered by NMS or assigned as match by gt")
    parser.add_argument('--m_th', type=float, default=0.35,
                        help="Probabilities higher than this threshold are marked as instances")
    parser.add_argument('--batch_size_test', type=int, default=64, help="Batch size during testing")
    parser.add_argument('--visualize', type=str2bool, default=False, help="True to see the images while testing")
    parser.add_argument('--view_grid', type=str2bool, default=False,
                        help="True to see the grid while testing. Only used if visualize=True or save_imgs=True")
    parser.add_argument('--save_imgs', type=str2bool, default=False,
                        help="True to save the images with the predictions")
    parser.add_argument('--max_boxes', type=int, default=20, help="Max number of boxes in an image during test."
                        "Necessary to have prior knowledge about the shape of the ground-truth tensor")

    # Files
    parser.add_argument('--dataset_name', type=str, default='PVDN/night', help="Name of the dataset")
    parser.add_argument('--dataset_path', type=str, default='/media/data/Datasets/Vehicles', help="Path to the dataset")
    parser.add_argument('--train_imgs', type=str, default='train.txt',
                        help="File containing a list of paths to train images")
    parser.add_argument('--test_imgs', type=str, default='test.txt',
                        help="File containing a list of paths to test images")
    parser.add_argument('--val_imgs', type=str, default='val.txt',
                        help="File containing a list of paths to validation images")
    parser.add_argument('--val_perc', type=float, default=0.2, help="Percentage of train images used to validate")
    parser.add_argument('--filter_classifiers', type=str2bool, default=False,
                        help="To reduce or not the number of classifiers based on a filter image")
    parser.add_argument('--filter_img', type=str, default='filter.png',
                        help="Apply a filter img to remove unnecessary classifiers. None if it is not desired")

    # Model
    parser.add_argument('--save_dir', type=str, default='models', help="Folder with the model weights of each training")
    parser.add_argument('--restore_model', type=str2bool, default=False, help="True to restore a model for training")
    parser.add_argument('--weights', type=str, default='weights_000.h5', help="Filename of trained model weights")
    parser.add_argument('--json_model', type=str, default='model_struct.json',
                        help="Json serialization of model structure")
    parser.add_argument('--num_layers', type=int, default=5, help="Controls the depth of the network")
    parser.add_argument('--rn_version', type=str, default='1',
                        help="Resnet architecture version: 1, 2 or keras")
    parser.add_argument('--classifier_conf', type=int, default=3,
                        help="Classifiers architecture version: 1, 2 , 3 or 4")  # Defined in resnet_models.py
    parser.add_argument('--resnet_keras_weights', type=str, default=None,
                        help="Path to pretrained weights for the ResNet version of Keras. Parameter only used if"
                             "rn_version=keras. Set it to None to use random initial weights. Set it to imagenet to"
                             "download pretrained weights on ImageNet.")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of classes to be detected")
    opts = parser.parse_args(args)
    opts.train_imgs = os.path.join(opts.dataset_path, opts.dataset_name, opts.train_imgs)
    opts.test_imgs = os.path.join(opts.dataset_path, opts.dataset_name, opts.test_imgs)
    opts.val_imgs = os.path.join(opts.dataset_path, opts.dataset_name, opts.val_imgs)
    opts.filter_img = os.path.join(opts.dataset_path, opts.dataset_name, opts.filter_img)

    # Check everything is correct
    assert opts.h_grid <= opts.img_width or opts.v_grid <= opts.img_height, "Grid size cannot be larger than image size"
    assert opts.h_grid > 0 or opts.v_grid > 0, "Grid sizes must be positive numbers"
    assert opts.batch_size > 0 or opts.batch_size_test > 0, "Batch size must be a positive number"
    assert 0 < opts.d_th <= np.min([opts.img_width, opts.img_height]), \
        "d_th must be in the range [0, max(img_width, img_height)]"
    assert 0 < opts.m_th <= 1, "m_th must be in the range [0, 1]"
    assert os.path.isdir(opts.dataset_path), "dataset_path does not exist"
    assert os.path.isfile(opts.train_imgs), "train_imgs does not exist"
    assert os.path.isfile(opts.test_imgs), "test_imgs does not exist"
    assert os.path.isfile(opts.val_imgs) or 0 <= opts.val_perc <= 1, \
        "Either val_imgs must exist or val_perc must be in the range [0, 1]"
    assert (opts.restore_model and os.path.isfile(os.path.join(opts.save_dir, opts.weights))) or \
           not opts.restore_model, "weights not exists"
    assert (opts.restore_model and os.path.isfile(os.path.join(opts.save_dir, opts.json_model))) or \
           not opts.restore_model, "json_model not exists"
    assert opts.rn_version in ['1', '2', 'keras'], "rn_version can only be 1, 2 or keras"
    assert opts.classifier_conf in range(1, 5), "classifier_conf can only be 1, 2, 3 or 4"
    assert opts.rn_version != 'keras' or (opts.rn_version == 'keras' and (os.path.isfile(opts.resnet_keras_weights) or
           opts.resnet_keras_weights == 'imagenet')), "If rn_version == keras, then resnet_keras_weights must be a" \
                                                      "valid path or it must be imagenet to download ImageNte weights"

    # Set seed
    set_seed(opts.seed)
    return opts






