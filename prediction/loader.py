import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pickle
import albumentations as albu
import glob
from recording.sequence_reader import SequenceReader
from tqdm import tqdm
import torch
import recording.util as util
import random
import prediction.pred_util as pred_util
import math
from prediction.augmentor import do_augmentation

# KEEP_P_C = {('9026', 0)}#, ('9027', 3)} # no shadow
# KEEP_P_C = {('9026', 3)}#, ('9027', 0)} # strong shadow


def visualize_batch(config, input_image, gt_force, batch_index=0, pred_force=None, show=True):
    # Visualizes a batch
    input_image = util.to_cpu_numpy(input_image)[batch_index, :].transpose(1, 2, 0)
    gt_force = util.to_cpu_numpy(gt_force)[batch_index, :]

    num_input_images = int(input_image.shape[2] / 3)
    num_subplots = 2 + num_input_images   # If we have a multichannel input image
    num_y = np.ceil(num_subplots / 2).astype(int)

    max_force = 1
    if config.FORCE_CLASSIFICATION:
        max_force = config.NUM_FORCE_CLASSES

    plt.subplot(num_y, 2, 1)
    plt.axis('off')
    plt.gca().set_title('GT Force')
    plt.imshow(gt_force.squeeze(), vmin=0, vmax=max_force)

    if pred_force is not None:
        pred_force = util.to_cpu_numpy(pred_force).squeeze()
        plt.subplot(num_y, 2, 2)
        plt.axis('off')
        plt.gca().set_title('Pred Force')
        plt.imshow(pred_force, vmin=0, vmax=max_force)

    for i in range(num_input_images):
        plt.subplot(num_y, 2, i + 3)
        plt.axis('off')
        plt.gca().set_title('Input image ' + str(i))
        plt.imshow(input_image[:, :, i*3:(i+1)*3])

    if show:
        plt.show()
    # plt.pause(0.02)
    # plt.draw()
    # plt.show()


class ForceDataset(Dataset):

    def __init__(
            self,
            config,
            seq_filter,
            augmentation=None,
            preprocessing_fn=None,
            image_method=0,
            force_method=False,
            skip_frames=1,
            randomize_cam_seq=True,
            include_raw_force=False,
            use_cameras=None,
            phase=None,
            bg_sub=False,
    ):
        self.config = config
        self.seq_filter = seq_filter
        self.skip_frames = skip_frames
        self.randomize_cam_seq = randomize_cam_seq
        self.use_cameras = use_cameras

        print('Loading dataset with filter: {}. Skipping {} frames'.format(seq_filter, skip_frames))
        self.all_datapoints = self.load_sequences(seq_filter, self.use_cameras)
        print('Done loading dataset. Total size:', len(self.all_datapoints))
        if len(self.all_datapoints) == 0:
            raise ValueError('Couldnt find datapoints')

        self.augmentation = augmentation
        self.image_method = image_method
        self.force_method = force_method
        self.preprocessing_fn = preprocessing_fn
        self.include_raw_force = include_raw_force
        self.phase = phase

    def __getitem__(self, i):
        # Any image/force pair has three parameters which uniquely define it. The timestep (frame #), which camera it
        # came from, and which sequence it came from. The pytorch dataloader wants to be able to access datapoints as
        # indexed by a single variable i, so given i, look in our lookup table which datapoint that should be.

        timestep = self.all_datapoints[i]['timestep']
        camera_idx = self.all_datapoints[i]['camera_idx']
        seq_reader = self.all_datapoints[i]['seq_reader']

        force_array = seq_reader.get_force_pytorch(camera_idx, timestep, self.config)
        image_0 = seq_reader.get_img_pytorch(camera_idx, timestep, self.config)

        image_list, force_array, _ = do_augmentation([image_0], force_array, seq_reader.sensel_points[camera_idx], self.config, self.phase)
        image_0 = image_list[0]
        raw_force_array = force_array

        if self.image_method == 0:  # Normal single image representation
            image_out = image_0
        elif self.image_method == 1:    # Four stacked images
            image_1 = seq_reader.get_img_pytorch(camera_idx, max(0, timestep - self.config.DATALOADER_MULTI_FRAME_STEP), self.config)
            image_2 = seq_reader.get_img_pytorch(camera_idx, max(0, timestep - self.config.DATALOADER_MULTI_FRAME_STEP * 2), self.config)
            image_3 = seq_reader.get_img_pytorch(camera_idx, max(0, timestep - self.config.DATALOADER_MULTI_FRAME_STEP * 3), self.config)
            image_out = np.concatenate((image_0, image_1, image_2, image_3), axis=2)
        elif self.image_method == 2:    # Difference frame
            image_1 = seq_reader.get_img_pytorch(camera_idx, max(0, timestep - self.config.DATALOADER_MULTI_FRAME_STEP), self.config)

            image = image_0 - image_1 + 0.5     # Center at 0.5
            image_out = np.clip(image, 0, 1)        # Clip to 0-1
        elif self.image_method == 3:    # Multiple frames
            image_out = np.zeros((self.config.DATALOADER_NUM_FRAMES, self.config.NETWORK_IMAGE_SIZE_Y, self.config.NETWORK_IMAGE_SIZE_X, 3))
            for image_idx in range(self.config.DATALOADER_NUM_FRAMES):
                t = max(0, timestep - self.config.DATALOADER_MULTI_FRAME_STEP * image_idx)
                f = seq_reader.get_img_pytorch(camera_idx, t, self.config)
                image_out[image_idx, :, :, :] = f

            image_out = image_out.transpose(3, 0, 1, 2)     # set to, 3, num_timesteps, y, x
        elif self.image_method == 4:    # Single image, but black and white
            image_out = cv2.cvtColor(image_0 * 255, cv2.COLOR_BGR2GRAY) / 255
            image_out = np.repeat(image_out[:, :, np.newaxis], 3, axis=2)
        elif self.image_method == 5:    # HALF RESOLUTION!!!
            image_out = cv2.resize(image_0, (0, 0), fx=0.5, fy=0.5)
            image_out = cv2.resize(image_out, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        elif self.image_method == 6:    # QUARTER RESOLUTION!!!
            image_out = cv2.resize(image_0, (0, 0), fx=0.25, fy=0.25)
            image_out = cv2.resize(image_out, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
        elif self.image_method == 7:    # EIGHTH RESOLUTION!!!
            image_out = cv2.resize(image_0, (0, 0), fx=0.125, fy=0.125)
            image_out = cv2.resize(image_out, (0, 0), fx=8.0, fy=8.0, interpolation=cv2.INTER_NEAREST)
        elif self.image_method == 8:    # SIXTEENTH RESOLUTION!!!
            image_out = cv2.resize(image_0, (0, 0), fx=0.0625, fy=0.0625)
            image_out = cv2.resize(image_out, (0, 0), fx=16.0, fy=16.0, interpolation=cv2.INTER_NEAREST)
        elif self.image_method == 9:    # THIRTYSECOND RESOLUTION!!!
            image_out = cv2.resize(image_0, (0, 0), fx=0.03125, fy=0.03125)
            image_out = cv2.resize(image_out, (0, 0), fx=32.0, fy=32.0, interpolation=cv2.INTER_NEAREST)

        if self.force_method == 0:  # Returns a scalar force representation
            force_array = np.clip(force_array / self.config.NORM_FORCE_REGRESS, 0, 1)
            force_array = np.expand_dims(force_array, axis=0)
        elif self.force_method == 1:    # Classes
            raise NotImplemented('This is no longer a valid option')
        elif self.force_method == 2:    # Classes with custom thresholds
            force_array = pred_util.scalar_to_classes(force_array, self.config.FORCE_THRESHOLDS)

        if self.preprocessing_fn is not None:
            image_out = self.preprocessing_fn(image_out)

        out_dict = dict()
        out_dict['img'] = self.to_tensor(image_out)
        out_dict['img_original'] = self.to_tensor(image_0)
        out_dict['force'] = force_array
        out_dict['seq_path'] = seq_reader.seq_path
        out_dict['camera_idx'] = camera_idx
        out_dict['timestep'] = timestep
        out_dict['participant'] = seq_reader.participant
        out_dict['action'] = seq_reader.action
        out_dict['raw_force'] = raw_force_array

        return out_dict

    def __len__(self):
        return len(self.all_datapoints)

    def to_tensor(self, x):
        if len(x.shape) == 3:   # image. Reformats into pytorch's way of taking images, which is (3,384,480)
            return x.transpose(2, 0, 1).astype('float32')
        elif len(x.shape) == 4:   # video
            return x.astype('float32')
        else:
            raise ValueError('Wrong number of channels')

    def load_sequences(self, seq_filter, use_cameras=None):
        """
        Takes a filter pointing to many dataset sequence and creates an list of "datapoints", each which is a training sample
        """
        if not isinstance(seq_filter, list):
            raise ValueError('Need a sequence filter list!')

        datapoints = []

        all_sequences = []
        for filter in seq_filter:
            all_sequences.extend(glob.glob(filter))

        for seq_path in all_sequences:
            if any([exclude in seq_path for exclude in self.config.EXCLUDE_ACTIONS]):
                continue

            seq_reader = SequenceReader(seq_path)
            for c in range(seq_reader.num_cameras):
                if use_cameras is not None and c not in use_cameras:
                    continue

                # ok = False
                # for s in KEEP_P_C:
                #     if s[0] in seq_reader.participant and s[1] == c:
                #         ok = True
                # if not ok:
                #     continue
                # print('keep', seq_reader.participant, c)

                this_camera_points = []
                for t in range(0, seq_reader.num_frames, self.skip_frames):
                    datapoint = dict()
                    datapoint['seq_reader'] = seq_reader
                    datapoint['camera_idx'] = c
                    datapoint['timestep'] = t
                    this_camera_points.append(datapoint)
                datapoints.append(this_camera_points)

        if self.randomize_cam_seq:
            random.shuffle(datapoints)

        flattened = [item for sublist in datapoints for item in sublist]

        return flattened


if __name__ == "__main__":
    from prediction.pred_util import *

    config = parse_config_args()
    train_dataset = ForceDataset(config, config.TEST_FILTER, image_method=config.DATALOADER_IMAGE_METHOD,
                                 # augmentation=get_training_augmentation(),
                                 force_method=config.DATALOADER_FORCE_METHOD,
                                 skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES)

    # test_dataset = ForceDataset(config, config.TEST_FILTER, image_method=config.DATALOADER_IMAGE_METHOD,
    #                             force_method=config.DATALOADER_FORCE_METHOD)
    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    video_data = []

    for idx, batch in enumerate(tqdm(test_dataloader)):
        image_model = batch['img']
        force_gt = batch['force']
        visualize_batch(config, image_model, force_gt)

        if idx > 300:
            break

