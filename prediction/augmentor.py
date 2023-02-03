import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pickle
import albumentations as albu
import glob
from tqdm import tqdm
import torch
import recording.util as util
import random
import prediction.pred_util as pred_util
import math
import albumentations


aug_albumen = albumentations.RandomBrightnessContrast()


def rotate_translate_crop(image, sensel_corners, output_shape, aug_rotate, aug_translate_x, aug_translate_y, aug_scale, get_coords=False):
    """
    Magic function to do combined rotation/translation/cropping
    :param image:  An image, can be RGB or force images
    :param sensel_corners: a numpy 2x4 list of sensel corners in image space
    :param output_shape: tuple, desired output image shape
    :param aug_rotate: rotation amount, degrees
    :param aug_translate_x: translation amount, x direction. +1.0 means sensel is half out of screen
    :param aug_translate_y: translation amount, y direction
    :param aug_scale: scaling. 1.0 is sensel fully occupies output image, larger is zoomed out
    :return:
    """

    # sensel_corners = sensel_corners.T # how did this happen?
    aug_rotate *= math.pi / 180

    # homogeneous transform matrices for each transformation
    translation = np.array([[1, 0, -image.shape[1] / 2], [0, 1, -image.shape[0] / 2], [0, 0, 1]], dtype='float32')
    rotation_matrix = np.array([[math.cos(aug_rotate), -math.sin(aug_rotate), 0], [math.sin(aug_rotate), math.cos(aug_rotate), 0], [0, 0, 1]], dtype='float32')
    undo_translation = np.array([[1, 0, image.shape[1] / 2], [0, 1, image.shape[0] / 2], [0, 0, 1]], dtype='float32')

    affine = undo_translation @ rotation_matrix @ translation   # applying transformations about center of image by translating before and after. Premultiply for global coords
    affine = affine[0:2, :]  # Put into OpenCV form

    rotated_image = cv2.warpAffine(image, affine, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)   # Warp image

    homogeneous_corners = np.vstack([sensel_corners.T, np.ones((1, 4))])
    rotated_sensel_corners = affine @ homogeneous_corners

    span_y, span_x, center_y, center_x = util.get_sensel_scale(rotated_sensel_corners.T, output_shape)

    center_x += aug_translate_x * span_x
    center_y += aug_translate_y * span_y
    span_x *= aug_scale
    span_y *= aug_scale

    min_x = max(int(center_x - span_x), 0)
    max_x = min(int(center_x + span_x), image.shape[1])
    min_y = max(int(center_y - span_y), 0)
    max_y = min(int(center_y + span_y), image.shape[0])

    if get_coords:
        return min_x, min_y, max_x, max_y

    # cropping to max and min of transformed points
    cropped_img = rotated_image[min_y:max_y, min_x:max_x]

    network_image_size = (output_shape[1], output_shape[0])   # Usually 480x384
    resized_img = cv2.resize(cropped_img, network_image_size)   # After cropping, resize to network size

    return resized_img


def do_augmentation(images, force, sensel_corners, config, phase, vis=False, first_frame=None):
    output_shape = (config.NETWORK_IMAGE_SIZE_Y, config.NETWORK_IMAGE_SIZE_X)

    if phase == 'train':
        rotation = random.uniform(-config.AUGMENT_ROTATION, config.AUGMENT_ROTATION)
        translate_x = random.uniform(-config.AUGMENT_TRANSLATION, config.AUGMENT_TRANSLATION)
        translate_y = random.uniform(-config.AUGMENT_TRANSLATION, config.AUGMENT_TRANSLATION)
        scale = random.uniform(config.AUGMENT_MIN_SCALE, config.AUGMENT_MAX_SCALE)

        aug_images = [rotate_translate_crop(i, sensel_corners, output_shape, rotation, translate_x, translate_y, scale) for i in images]
        aug_force = rotate_translate_crop(force, sensel_corners, output_shape, rotation, translate_x, translate_y, scale)

        if first_frame is not None:
            first_frame = rotate_translate_crop(first_frame, sensel_corners, output_shape, rotation, translate_x, translate_y, scale)

        if config.AUGMENT_FLIP and random.getrandbits(1):
            aug_images = [cv2.flip(i, 0) for i in aug_images]
            aug_force = cv2.flip(aug_force, 0)

        if config.AUGMENT_BRIGHTNESS_CONTRAST:
            aug_images[0] = aug_albumen.apply(aug_images[0])
            if len(aug_images) > 1:
                print('Currently doesnt support videos')

        if vis:
            plt.subplot(2, 1, 1)
            plt.imshow(images[0])
            plt.subplot(2, 1, 2)
            plt.imshow(aug_images[0])
            plt.show()
    else:
        aug_images = [rotate_translate_crop(i, sensel_corners, output_shape, 0, 0, 0, config.VAL_SCALE) for i in images]
        aug_force = rotate_translate_crop(force, sensel_corners, output_shape, 0, 0, 0, config.VAL_SCALE)

    return aug_images, aug_force, first_frame


if __name__ == "__main__":
    from prediction.pred_util import *
    from prediction.loader import ForceDataset, visualize_batch

    config = parse_config_args()
    train_dataset = ForceDataset(config, config.VAL_FILTER, image_method=config.DATALOADER_IMAGE_METHOD,
                                 force_method=config.DATALOADER_FORCE_METHOD,
                                 skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES, phase='val')

    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    video_data = []

    for idx, batch in enumerate(tqdm(test_dataloader)):
        image_model = batch['img']
        force_gt = batch['force']
        visualize_batch(config, image_model, force_gt)

        if idx > 300:
            break

