import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import torch
import numpy as np
from prediction.loader import ForceDataset
from torch.utils.data import DataLoader
import os
import prediction.trainer
from tqdm import tqdm
from prediction.model_builder import build_model
from prediction.pred_util import *
import random
import torch.multiprocessing
from recording.util import *
torch.multiprocessing.set_sharing_strategy('file_system')


def test(num_frames=8000, overlay=False, draw_lines=False):
    config.DATALOADER_TEST_SKIP_FRAMES = 1
    # config.VAL_FILTER = ['data/train/*/*']

    random.seed(5)  # Set the seed so the sequences will be randomized the same
    model_dict = build_model(config, device, ['val'])

    best_model = model_dict['model']
    best_model.load_state_dict(torch.load(find_latest_checkpoint(config)))
    best_model.eval()

    val_dataloader = DataLoader(model_dict['val_dataset'], batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    # val_dataloader = DataLoader(model_dict['train_dataset'], batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    out_path = os.path.join('data', 'movies', config.CONFIG_NAME + '_movie.avi')
    print('Saving to:', out_path)
    mkdir(out_path, cut_filename=True)
    mw = MovieWriter(out_path, fps=30)

    for idx, batch in enumerate(tqdm(val_dataloader)):
        image_model = batch['img']
        image_original = batch['img_original']
        force_gt = batch['raw_force']

        with torch.no_grad():
            if config.FORCE_CLASSIFICATION:
                force_pred_class = best_model(image_model.cuda())
                force_pred_class = torch.argmax(force_pred_class, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = best_model(image_model.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS

            image_save = image_original.detach().squeeze().cpu().numpy().transpose((1, 2, 0))
            image_save = cv2.cvtColor(image_save * 255, cv2.COLOR_BGR2RGB).astype(np.uint8)
            participant = batch['participant'][0]
            action = batch['action'][0]

            force_pred_scalar[force_pred_scalar < 0] = 0    # Clip in case of negative values
            force_color_gt = pressure_to_colormap(force_gt.detach().squeeze().cpu().numpy())
            force_color_pred = pressure_to_colormap(force_pred_scalar.detach().squeeze().cpu().numpy())

            if overlay:
                val_img = 0.3
                force_color_gt = cv2.addWeighted(force_color_gt, 1.0, image_save, val_img, 0)
                force_color_pred = cv2.addWeighted(force_color_pred, 1.0, image_save, val_img, 0)

            cv2.putText(image_save, 'RGB', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(image_save, '{} {}'.format(participant, batch['timestep'][0].item()), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(image_save, '{}'.format(action), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(force_color_gt, 'GT Force', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(force_color_pred, 'Predicted Force', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # img_network = np.zeros_like(image_save)
            img_network = image_model.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            img_network = (resnet_invert_preprocess(img_network) * 255).astype(np.uint8)
            img_network = cv2.cvtColor(img_network, cv2.COLOR_BGR2RGB)

            combined_frame_top = np.concatenate((image_save, force_color_gt), axis=1)
            combined_frame_bot = np.concatenate((img_network, force_color_pred), axis=1)
            combined_frame = np.concatenate((combined_frame_top, combined_frame_bot), axis=0)
            mw.write_frame(combined_frame)

        if idx > num_frames:
            break

    mw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=8000)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--draw_lines', action='store_true')
    parser.add_argument('-cfg', '--config', type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test(args.frames, overlay=args.overlay, draw_lines=args.draw_lines)
