import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import os
from prediction.augmentor import rotate_translate_crop
from recording.util import *
import json
from functools import lru_cache


class SequenceReader:
    def __init__(self, seq_path):
        self.seq_path = seq_path

        split_path = os.path.normpath(seq_path).split(os.path.sep)
        self.action = split_path[-1]
        self.participant = split_path[-2]
        # self.has_left, self.has_right = self.sequence_to_handedness(seq_path)

        self.metadata = json_read(os.path.join(self.seq_path, 'metadata.json'))
        self.num_cameras = self.metadata['num_cameras']
        self.num_frames = self.metadata['num_frames']
        self.timesteps = self.metadata['timesteps']
        # self.lighting = self.get_lighting()
        # self.skin_tone = self.metadata['participant']['Skintone']

        # self.force_sum = self.get_force_time()    # slow AF
        self.img_height = []
        self.img_width = []
        for i in range(self.num_cameras):
            # sample_img = self.get_img(i, 0)
            # self.img_height.append(sample_img.shape[0])
            # self.img_width.append(sample_img.shape[1])
            self.img_height.append(1080)
            self.img_width.append(1920)

        self.sensel_homography = [self.get_sensel_homography(c)[0] for c in range(self.num_cameras)]
        self.sensel_points = [self.get_sensel_homography(c)[1] for c in range(self.num_cameras)]

        self.memoize_pixel_size_arr = [None] * self.num_cameras
        self.memoize_first_img = dict()

    def get_lighting(self):
        long_str = self.metadata['participant']['Light Direction']

        out_str = ''
        if 'left' in long_str:
            out_str += 'L'
        if 'center' in long_str:
            out_str += 'C'
        if 'right' in long_str:
            out_str += 'R'

        # if len(out_str) == 0:
        #     raise ValueError('No lighting condition found')

        return out_str

    def sequence_to_handedness(self, seq):
        special_cases = {   # Left, right
            'type_sentence_5x_left': (True, True),
            'type_sentence_5x_right': (True, True),
            'type_sentence_5x_both': (True, True),
            'type_sentence_5x_both(2)': (True, True),
            'type_ipad_5x_both': (True, False),         # LEFT
            'type_ipad_5x_both(2)': (False, True),         # RIGHT
        }

        parts = os.path.normpath(seq).split(os.sep)     # Split to get the individual path parts
        seq_name = parts[-1]
        # seq_name = '_'.join(seq_name.split('_')[:-1])

        for s in special_cases:
            if seq_name == s:
                return special_cases[seq_name]

        if 'left' in seq_name:
            return True, False
        elif 'right' in seq_name:
            return False, True
        elif 'left' in seq:
            return True, False
        elif 'right' in seq:
            return False, True
        else:
            raise Exception('No handedness found')

    def get_force_time(self):
        # Gets the sum of force across time
        out = []
        for t in range(self.num_frames):
            force = self.get_pressure_kPa(t)
            out.append(force.sum())

        out = np.array(out)
        return out

    def get_img(self, camera_idx, frame_idx, to_rgb=False):
        img = cv2.imread(self.get_img_path(camera_idx, frame_idx))
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # def crop_img(self, img, camera_idx, config):
    #     print('Shouldnt be using this crop function')
    #     min_xy = self.sensel_points[camera_idx].min(axis=0) - config.CROP_MARGIN    # These are XY
    #     max_xy = self.sensel_points[camera_idx].max(axis=0) + config.CROP_MARGIN
    #     center_xy = np.round((min_xy + max_xy) / 2).astype(int)
    #
    #     # img_radius = np.round((max_xy - min_xy).max() / 2).astype(int)
    #     span_x = (max_xy[0] - min_xy[0]) / 2
    #     span_y = (max_xy[1] - min_xy[1]) / 2
    #     if span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X > span_y:     # Y is smaller
    #         span_y = span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X     # set Y to be based on X
    #     else:
    #         span_x = span_y * config.NETWORK_IMAGE_SIZE_X / config.NETWORK_IMAGE_SIZE_Y
    #
    #     min_x = max(int(center_xy[0] - span_x), 0)
    #     max_x = min(int(center_xy[0] + span_x), img.shape[1])
    #     min_y = max(int(center_xy[1] - span_y), 0)
    #     max_y = min(int(center_xy[1] + span_y), img.shape[0])
    #
    #     network_image_size = (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y)
    #     out_img = cv2.resize(img[min_y:max_y, min_x:max_x, ...], network_image_size)    # image is YX
    #
    #     return out_img

    def get_force_pytorch(self, camera_idx, frame_idx, foo=None):
        force = self.get_force_warped_to_img(camera_idx, frame_idx).astype('float32')

        # if config.CROP_IMAGES is True:
        #     force = self.crop_img(force, camera_idx, config)
        return force

    def get_img_pytorch(self, camera_idx, frame_idx, foo=None):
        """
        Helper function to get images in a pytorch-friendly format
        """
        img = self.get_img(camera_idx, frame_idx, to_rgb=True).astype('float32') / 255

        # if config.CROP_IMAGES is True:
        #     img = self.crop_img(img, camera_idx, config)
        return img

    def get_first_img_intensity(self, camera_idx):
        if camera_idx not in self.memoize_first_img:
            img = self.get_img_pytorch(camera_idx, 0, None)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # self.memoize_first_img[camera_idx] = img
            return img

        return self.memoize_first_img[camera_idx]

    def get_depth(self, frame_idx):
        img_path = os.path.join(self.seq_path, 'depth', '{:05d}.png'.format(frame_idx))
        img = cv2.imread(img_path, -1)

        return img / 1000.0     # Convert back into meters

    def get_ir(self, frame_idx):
        pkl_path = os.path.join(self.seq_path, 'ir', '{:05d}.pkl'.format(frame_idx))

        with open(pkl_path, 'rb') as handle:
            ir_pkl = pickle.load(handle)

        return ir_pkl['img']

    def get_img_path(self, camera_idx, frame_idx):
        return os.path.join(self.seq_path, 'camera_{}'.format(camera_idx), '{:05d}.jpg'.format(frame_idx))

    def get_pressure_kPa(self, frame_idx):
        pkl_path = os.path.join(self.seq_path, 'force', '{:05d}.pkl'.format(frame_idx))
        with open(pkl_path, 'rb') as handle:
            raw_counts = pickle.load(handle)

        kPa = convert_counts_to_kPa(raw_counts)
        return kPa

    def get_camera_params(self, camera_idx):
        params = dict()
        c = self.metadata['camera_calibrations'][str(camera_idx)]
        params['extrinsic'] = np.array(c['ModelViewMatrix'])
        params['intrinsic'] = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]])
        params['distortion'] = np.array([c['k1'], c['k2'], c['p1'], c['p2'], c['k3'], c['k4'], c['k5'], c['k6']])
        return params

    def get_sensel_homography(self, camera_idx):
        """
        Gets the 3x3 homography matrix to transform a point in sensel space into image space
        """
        if 'MYCAGE' in self.metadata:
            # handle my cage
            if 'camera_calibrations' not in self.metadata:
                return None, None

            homography = np.array(self.metadata['camera_calibrations'][str(camera_idx)]['homography'])
            imgpts = np.array(self.metadata['camera_calibrations'][str(camera_idx)]['imgpts'])
            return homography, imgpts

        camera_params = self.get_camera_params(camera_idx)

        rvec, jacobian = cv2.Rodrigues(camera_params['extrinsic'][:3, :3])
        tvec = camera_params['extrinsic'][:3, 3] / 1000.0  # Convert to meters

        sensel_w = 0.235
        sensel_h = 0.135
        sensel_origin_x = 0.016
        sensel_origin_y = 0.014
        sensel_z = -0.001

        sensel_corners_2D = np.float32([[185, 0], [185, 105], [0, 105], [0, 0]])
        sensel_corners_3D = np.float32([[sensel_origin_y + sensel_h, sensel_z, sensel_origin_x + sensel_w],
                                        [sensel_origin_y, sensel_z, sensel_origin_x + sensel_w],
                                        [sensel_origin_y, sensel_z, sensel_origin_x],
                                        [sensel_origin_y + sensel_h, sensel_z, sensel_origin_x]])

        points = np.array(sensel_corners_3D)
        image_points, jacobian = cv2.projectPoints(points, rvec, tvec, camera_params['intrinsic'], np.array([]))

        homography, status = cv2.findHomography(sensel_corners_2D, image_points[:, 0, :])
        return homography, image_points[:, 0, :]

    def get_force_warped_to_img(self, camera_idx, frame_idx, draw_sensel=False, manual_force_input=None):
        if manual_force_input is not None:
            force_img = manual_force_input
        else:
            force_img = self.get_pressure_kPa(frame_idx)

        force_warped = cv2.warpPerspective(force_img, self.sensel_homography[camera_idx], (self.img_width[camera_idx], self.img_height[camera_idx]))

        if draw_sensel:
            for c_idx in range(4):  # Draw the four corners on the image
                start_point = tuple(self.sensel_points[camera_idx][c_idx, :].astype(int))
                end_point = tuple(self.sensel_points[camera_idx][(c_idx + 1) % 4, :].astype(int))
                cv2.line(force_warped, start_point, end_point, 2, 5)

        return force_warped

    def get_force_cropped_pressure_img(self, camera_idx, config, pressure_img_space, frame_idx=None):
        """
        Calculates the force in a pressure image, already cropped
        """
        img_space_sq_meters_per_pixel = self.get_pixel_size_arr(camera_idx, config)
        total_force_img = (pressure_img_space * 1000 * img_space_sq_meters_per_pixel).sum()

        if frame_idx is not None:
            pressure_sensel_space_gt = self.get_pressure_kPa(frame_idx)
            total_force_orig = pressure_sensel_space_gt.sum() * 1000 * 0.00125 * 0.00125
            print(f'gt force {total_force_orig}, pred img unwarp {total_force_img}')

        return total_force_img

    def get_pixel_size_arr(self, camera_idx, config):
        if self.memoize_pixel_size_arr[camera_idx] is not None:
            return self.memoize_pixel_size_arr[camera_idx]

        # MAGIC function to get a "square meters per pixel" array given a camera angle and cropping. This is used for finding newtons
        calc_pitch = 20
        sensel_pixel_size = 0.00125 * calc_pitch
        x_range = np.arange(0-50, 185+50, calc_pitch)
        y_range = np.arange(0-50, 105+50, calc_pitch)
        xx, yy = np.meshgrid(x_range, y_range)

        points_sensel_space = np.zeros((xx.size, 2))
        points_sensel_space[:, 1] = xx.flatten()
        points_sensel_space[:, 0] = yy.flatten()

        points_camera_space = apply_homography(points_sensel_space, self.sensel_homography[camera_idx]) # now this is in YX!!

        # plt.imshow(self.get_img(camera_idx, 0, to_rgb=True))
        # plt.scatter(points_camera_space[:, 1], points_camera_space[:, 0])
        # plt.show()

        points_camera_space = np.flip(points_camera_space, axis=1)  # turn back to xy. do I know what's going on? no

        # # now do cropping
        # min_xy = self.sensel_points[camera_idx].min(axis=0) - config.CROP_MARGIN    # These are XY
        # max_xy = self.sensel_points[camera_idx].max(axis=0) + config.CROP_MARGIN
        # center_xy = np.round((min_xy + max_xy) / 2).astype(int)
        #
        # # img_radius = np.round((max_xy - min_xy).max() / 2).astype(int)
        # span_x = (max_xy[0] - min_xy[0]) / 2
        # span_y = (max_xy[1] - min_xy[1]) / 2
        # if span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X > span_y:     # Y is smaller
        #     span_y = span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X     # set Y to be based on X
        # else:
        #     span_x = span_y * config.NETWORK_IMAGE_SIZE_X / config.NETWORK_IMAGE_SIZE_Y
        #
        # min_x = int(center_xy[0] - span_x)
        # max_x = int(center_xy[0] + span_x)
        # min_y = int(center_xy[1] - span_y)
        # max_y = int(center_xy[1] + span_y)

        sensel_corners = self.sensel_points[camera_idx]

        rotated_sensel_corners = np.vstack([sensel_corners.T])

        span_y, span_x, center_y, center_x = get_sensel_scale(rotated_sensel_corners.T, (config.NETWORK_IMAGE_SIZE_Y, config.NETWORK_IMAGE_SIZE_X))

        center_x += 0 * span_x
        center_y += 0 * span_y
        span_x *= config.VAL_SCALE
        span_y *= config.VAL_SCALE

        min_x = int(center_x - span_x)
        max_x = int(center_x + span_x)
        min_y = int(center_y - span_y)
        max_y = int(center_y + span_y)

        network_image_size = (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y)

        points_cropped = points_camera_space.copy()
        points_cropped[:, 0] = (points_camera_space[:, 0] - min_x) * network_image_size[0] / (max_x - min_x)
        points_cropped[:, 1] = (points_camera_space[:, 1] - min_y) * network_image_size[1] / (max_y - min_y)

        points_cropped_grid = points_cropped.reshape(xx.shape[0], xx.shape[1], 2)
        points_cropped_grid_noedge = points_cropped_grid[:-1, :-1, :]

        sq_meters_per_pixel = np.zeros((points_cropped_grid.shape[0] - 1, points_cropped_grid.shape[1] - 1))
        for x in range(sq_meters_per_pixel.shape[0]):
            for y in range(sq_meters_per_pixel.shape[1]):
                diff_x = points_cropped_grid[x, y, :] - points_cropped_grid[x+1, y, :]
                diff_y = points_cropped_grid[x, y, :] - points_cropped_grid[x, y+1, :]
                pixel_area_cell = np.abs(np.cross(diff_x, diff_y))
                meters_cell = sensel_pixel_size * sensel_pixel_size
                sq_meters_per_pixel[x,y] = meters_cell / pixel_area_cell

        # plt.clf()
        # plt.imshow(self.crop_img(self.get_img(camera_idx, 0, to_rgb=True), camera_idx, config))
        # plt.scatter(points_cropped_grid_noedge[:, :, 0].flatten(), points_cropped_grid_noedge[:, :, 1].flatten(), c=sq_meters_per_pixel.flatten())
        # plt.colorbar()
        # plt.show()

        points_cropped_grid_noedge_flat = points_cropped_grid_noedge.reshape(-1, 2)
        sq_meters_per_pixel_flat = sq_meters_per_pixel.reshape(-1)

        img_space_sq_meters_per_pixel = np.zeros((config.NETWORK_IMAGE_SIZE_Y, config.NETWORK_IMAGE_SIZE_X))
        for x in range(img_space_sq_meters_per_pixel.shape[1]):
            for y in range(img_space_sq_meters_per_pixel.shape[0]):
                this_coord = np.array([x, y])
                dist_to_grid = np.linalg.norm(this_coord - points_cropped_grid_noedge_flat, axis=1)
                min_idx = np.argmin(dist_to_grid)
                img_space_sq_meters_per_pixel[y, x] = sq_meters_per_pixel_flat[min_idx]

        # plt.clf()
        # plt.imshow(img_space_sq_meters_per_pixel)
        # plt.colorbar()
        # plt.show()
        self.memoize_pixel_size_arr[camera_idx] = img_space_sq_meters_per_pixel
        return img_space_sq_meters_per_pixel

        # frame_idx = 47
        # press_img = self.get_pressure_kPa(frame_idx)
        # total_force_orig = press_img.sum() * 1000 * 0.00125 * 0.00125
        #
        #
        # press_img_space = self.get_force_warped_to_img(camera_idx, frame_idx)
        # press_img_space = self.crop_img(press_img_space, camera_idx, config)
        # total_force_img = (press_img_space * 1000 * img_space_sq_meters_per_pixel).sum()
        # print(total_force_orig, total_force_img)
        # print('sup')
        # pass

    def get_force_overlay_img(self, camera_idx, frame_idx, colormap=cv2.COLORMAP_INFERNO):
        force_warped = self.get_force_warped_to_img(camera_idx, frame_idx, draw_sensel=False)
        force_color_warped = pressure_to_colormap(force_warped, colormap=colormap)

        img = self.get_img(camera_idx, frame_idx)

        return cv2.addWeighted(img, 1.0, force_color_warped, 1.0, 0.0)

    def get_overall_frame(self, frame_idx, config, overlay_force=True):
        """
        Returns a frame with all views and cameras rendered as subwindows
        :return: A numpy array
        """
        # out_x = 1440  # Rendering X, y
        # out_y = 810

        out_x = 480 * 2  # Rendering X, y
        out_y = 384 * 2

        panels_x = 2
        panelx_y = 2
        inc_x = out_x // panels_x
        inc_y = out_y // panelx_y

        def set_subframe(subframe_id, frame, val, title=None):
            if title is not None:
                cv2.putText(val, str(title), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            start_x = (subframe_id % panels_x) * inc_x
            start_y = (subframe_id // panels_x) * inc_y
            frame[start_y:start_y + inc_y, start_x:start_x + inc_x] = val

        cur_frame = np.zeros((out_y, out_x, 3))

        # force = self.get_pressure_kPa(frame_idx)
        # set_subframe(0, cur_frame, cv2.resize(pressure_to_colormap(force), (inc_x, inc_y)) / 255.0, 'Force')

        # ir = np.clip(self.get_ir(frame_idx) / 4000, 0, 1) * 255
        # ir_color = cv2.applyColorMap(ir.astype(np.uint8), cv2.COLORMAP_BONE)
        # set_subframe(1, cur_frame, cv2.resize(ir_color, (inc_x, inc_y)) / 255.0, 'IR')

        # depth = np.clip(self.get_depth(frame_idx) / 1.5, 0, 1) * 255
        # depth_color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_HOT)
        # set_subframe(1, cur_frame, cv2.resize(depth_color, (inc_x, inc_y)) / 255.0, 'Depth')

        for c in range(self.num_cameras):
            if overlay_force:
                img = self.get_force_overlay_img(c, frame_idx)
                img = rotate_translate_crop(img, self.sensel_points[c], (config.NETWORK_IMAGE_SIZE_Y, config.NETWORK_IMAGE_SIZE_X), 0, 0, 0, config.VAL_SCALE)
            else:
                img = self.get_img(c, frame_idx)
                img = rotate_translate_crop(img, self.sensel_points[c], (config.NETWORK_IMAGE_SIZE_Y, config.NETWORK_IMAGE_SIZE_X), 0, 0, 0, config.VAL_SCALE)

            set_subframe(c + 0, cur_frame, cv2.resize(img, (inc_x, inc_y)) / 255.0, 'Cam{}'.format(c))

        return cur_frame
