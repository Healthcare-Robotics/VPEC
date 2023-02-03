import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import os
import json
from pathlib import Path
import torch


SENSEL_COUNTS_TO_NEWTON = 1736   # How many force counts are approximately equal to one gram
SENSEL_PIXEL_PITCH = 0.00125    # The size of each force pixel
SENSEL_MAX_VIS = 20


def get_sensel_scale(sensel_corners, output_shape):
    # Magic function to get the

    min_xy = sensel_corners.min(axis=0)  # These are XY
    max_xy = sensel_corners.max(axis=0)

    center_xy = np.round((min_xy + max_xy) / 2).astype(int)
    span_x = (max_xy[0] - min_xy[0]) / 2
    span_y = (max_xy[1] - min_xy[1]) / 2

    aspect_ratio = output_shape[1] / output_shape[0]  # X divided by Y
    if span_x / aspect_ratio > span_y:  # X is bigger, need to inflate Y
        span_y = span_x / aspect_ratio  # set Y to be based on X
    else:
        span_x = span_y * aspect_ratio

    return span_y, span_x, center_xy[1], center_xy[0]


def colorize(image, clipping_range=(None, None), colormap=cv2.COLORMAP_HSV):
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def set_subframe(subframe_id, subframe, frame, steps_x=2, steps_y=2, title=None):
    """ Helper function when making a large image up of many tiled smaller images"""
    frame_x = frame.shape[1]
    frame_y = frame.shape[0]
    inc_x = frame_x // steps_x
    inc_y = frame_y // steps_y

    subframe = cv2.resize(subframe, (inc_x, inc_y), cv2.INTER_NEAREST) / 255.0

    if title is not None:
        cv2.putText(subframe, str(title), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(subframe, str(title), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    start_x = (subframe_id % steps_x) * inc_x
    start_y = (subframe_id // steps_x) * inc_y
    frame[start_y:start_y + inc_y, start_x:start_x + inc_x] = subframe


def pressure_to_colormap(kPa, log_scaling=False, colormap=cv2.COLORMAP_INFERNO):
    # Rescale the force array and apply the colormap
    # print('kPa max', kPa.max())

    if log_scaling:
        pressure_array = np.log1p(kPa) / np.log1p(SENSEL_MAX_VIS) * 255     # log scaling
    else:
        pressure_array = kPa * (255.0 / SENSEL_MAX_VIS) # linear scaling

    pressure_array[pressure_array > 255] = 255

    # force_color = cv2.applyColorMap(pressure_array.astype(np.uint8), cv2.COLORMAP_OCEAN)
    force_color = cv2.applyColorMap(pressure_array.astype(np.uint8), colormap)
    return force_color


def convert_counts_to_newtons(input_array):
    return input_array / SENSEL_COUNTS_TO_NEWTON


def convert_counts_to_kPa(input_array):
    # convert to kilopascals
    force = convert_counts_to_newtons(input_array)
    pa = force / (SENSEL_PIXEL_PITCH ** 2)
    return pa / 1000


def convert_kPa_to_newtons(kPa):
    return kPa * 1000 * (SENSEL_PIXEL_PITCH ** 2)


def mkdir(path, cut_filename=False):
    if cut_filename:
        path = os.path.dirname(os.path.abspath(path))
    Path(path).mkdir(parents=True, exist_ok=True)


def pkl_read(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def pkl_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(os.path.dirname(path))

    with open(path, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def json_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(path, cut_filename=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_read(path):
    with open(path, 'rb') as f:
        return json.load(f)


def apply_homography(points, h, yx=True):
    # Apply 3x3 homography matrix to points
    # Note that the homography matrix is parameterized as XY,
    # but all image coordinates are YX

    if yx:
        points = np.flip(points, 1)

    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    tform_h = np.matmul(h, points_h.T).T
    tform_h = tform_h / tform_h[:, 2][:, np.newaxis]

    points = tform_h[:, :2]

    if yx:
        points = np.flip(points, 1)

    return points


class MovieWriter:
    def __init__(self, path, fps=30):
        self.writer = None
        self.path = path
        self.fps = fps

    def write_frame(self, frame):
        if self.writer is None:
            mkdir(self.path, cut_filename=True)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def close(self):
        self.writer.release()


def np_find_nearest(a, value):
    """
    Find the index of the closest value in a numy array
    :param a: numpy array
    :param value: Query value
    :return:
    """
    a = np.asarray(a)
    idx = (np.abs(a - value)).argmin()
    return int(idx)


def to_cpu_numpy(obj):
    """Convert torch cuda tensors to cpu, numpy tensors"""
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_cpu_numpy(v)
            return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_cpu_numpy(v))
        return res
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise TypeError("Invalid type for move_to")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)