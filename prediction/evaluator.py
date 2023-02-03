import os.path

import torch
import numpy as np
from prediction.loader import *
import prediction.trainer
from tqdm import tqdm
import recording.util as util
from prediction.pred_util import *
from prediction.model_builder import build_model
import torchmetrics
import pprint
import datetime
from recording.util import json_write
import torch.nn.functional as F


class VolumetricIOU(torchmetrics.Metric):
    """
    This calculates the IoU summed over the entire dataset, then averaged. This means an image with no
    GT or pred force will contribute none to this metric.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape

        high = torch.maximum(preds, target)
        low = torch.minimum(preds, target)

        self.numerator += torch.sum(low)
        self.denominator += torch.sum(high)

    def compute(self):
        return self.numerator / self.denominator


class ContactIOU(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape
        assert preds.dtype == torch.long    # Make sure we're getting ints

        bool_pred = preds > 0
        bool_gt = target > 0

        self.numerator += torch.sum(bool_gt & bool_pred)
        self.denominator += torch.sum(bool_gt | bool_pred)

    def compute(self):
        return self.numerator / self.denominator


def reset_metrics(all_metrics):
    for key, metric in all_metrics.items():
        metric.reset()


def print_metrics(all_metrics, config, network_name='', save=True):
    out_dict = dict()
    for key, metric in all_metrics.items():
        val = metric.compute().item()
        print(key, val)
        out_dict[key] = val

    if save:
        d = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_filename = os.path.join('data', 'eval', f"{os.path.basename(network_name)}_{d}.txt")
        json_write(out_filename, out_dict, auto_mkdir=True)


CONTACT_THRESH = 1.0


def run_metrics(all_metrics, pressure_gt, pressure_pred, config):
    # Takes CUDA BATCHES as input
    pressure_pred = pressure_pred.detach()  # just in case

    contact_pred = (pressure_pred > CONTACT_THRESH).long()
    contact_gt = (pressure_gt > CONTACT_THRESH).long()

    all_metrics['contact_iou'](contact_pred, contact_gt)

    all_metrics['mse'](pressure_pred, pressure_gt)
    all_metrics['mae'](pressure_pred, pressure_gt)
    all_metrics['vol_iou'](pressure_pred, pressure_gt)

    any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
    any_contact_gt = torch.sum(contact_gt, dim=(1, 2)) > 0

    all_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)


def setup_metrics(device):
    all_metrics = dict()

    all_metrics['contact_iou'] = ContactIOU().to(device)
    all_metrics['mse'] = torchmetrics.MeanSquaredError().to(device)
    all_metrics['mae'] = torchmetrics.MeanAbsoluteError().to(device)
    all_metrics['vol_iou'] = VolumetricIOU().to(device)
    all_metrics['temporal_accuracy'] = torchmetrics.Accuracy(task='binary').to(device)
    return all_metrics


def evaluate(config, device):
    config.DATALOADER_TEST_SKIP_FRAMES = 1
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 12

    print('RUNNING EVAL, SKIPPING', config.DATALOADER_TEST_SKIP_FRAMES)
    random.seed(5)  # Set the seed so the sequences will be randomized the same
    model_dict = build_model(config, device, ['val'])

    checkpoint_path = find_latest_checkpoint(config)
    best_model = model_dict['model']
    best_model.load_state_dict(torch.load(checkpoint_path))
    best_model.eval()

    val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    tot_samples = 0

    all_metrics = setup_metrics(device)

    for idx, batch in enumerate(tqdm(val_loader)):
        batch_size = len(batch['participant'])
        with torch.no_grad():
            image = batch['img']

            if config.FORCE_CLASSIFICATION:
                force_pred_class = best_model(image.cuda())
                force_pred_class = torch.argmax(force_pred_class, dim=1)
                force_pred_scalar = pred_util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = best_model(image.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS
                F.relu(force_pred_scalar, inplace=True)

            force_gt_scalar = batch['raw_force'].cuda()

            run_metrics(all_metrics, force_gt_scalar, force_pred_scalar, config)

            tot_samples += batch_size

    print_metrics(all_metrics, config, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate(config, device)
