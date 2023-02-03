import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import recording.util as util
import argparse
import datetime
import os
import yaml
import argparse
from types import SimpleNamespace
import glob

CONFIG_BASE_PATH = './config'


def scalar_to_classes(scalar, thresholds):
    """
    Bins a float scalar into integer class indices. Could be faster, but is hopefully readable!
    :param scalar: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :return:
    """
    if torch.is_tensor(scalar):
        # out = -torch.ones_like(scalar, dtype=torch.int64)
        out = torch.zeros_like(scalar, dtype=torch.int64)
    else:
        # out = -np.ones_like(scalar, dtype=np.int64)
        out = np.zeros_like(scalar, dtype=np.int64)

    for idx, threshold in enumerate(thresholds):
        out[scalar >= threshold] = idx  # may overwrite the same value many times

    # if out.min() < 0:
    #     raise ValueError('Thresholds were not broad enough')

    return out


def classes_to_scalar(classes, thresholds):
    """
    Converts an integer class array into floating values. Obviously some discretization loss here
    :param classes: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :param final_value: if greater than the last threshold, fill in with this value
    :return:
    """
    if torch.is_tensor(classes):    # fill with negative ones
        out = -torch.ones_like(classes, dtype=torch.float)
    else:
        out = -np.ones_like(classes, dtype=np.float)

    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            val = thresholds[0]
        elif idx == len(thresholds) - 1:
            final_value = thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2    # Set it equal to the last value, plus half to gap to the previous thresh
            val = final_value
        else:
            val = (thresholds[idx] + thresholds[idx + 1]) / 2

        out[classes == idx] = val

    if out.min() < 0:
        raise ValueError('Thresholds were not broad enough')

    return out


def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    return load_config(args.config)


def load_config(config_name):
    config_path = os.path.join(CONFIG_BASE_PATH, config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    # data_obj = namedtuple('MyTuple', data)
    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj


def find_latest_checkpoint(config):
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    """
    all_checkpoints = glob.glob('data/model/*.pt')
    possible_matches = []
    for p in all_checkpoints:
        f = os.path.basename(p)
        if not f.startswith(config.CONFIG_NAME):
            continue
        f = f[len(config.CONFIG_NAME):-4] # cut off the prefix and suffix
        if not f.lower().islower():     # if it has any letters
            possible_matches.append(p)

    if len(possible_matches) == 0:
        raise ValueError('No valid model checkpoint files found')

    latest_file = max(possible_matches, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    return latest_file


def resnet_preprocessor(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if rgb.shape[2] == 12:
        mean = mean.repeat(4)
        std = std.repeat(4)

    rgb = rgb - mean
    rgb = rgb / std
    return rgb


def resnet_invert_preprocess(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # rgb = rgb - mean
    rgb = rgb * std + mean
    return rgb


def i3d_rgb_preprocessor(rgb):
    rgb = (rgb * 2) - 1
    return rgb




