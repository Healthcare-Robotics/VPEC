import torch
import numpy as np
import segmentation_models_pytorch as smp
from prediction.pred_util import *

import ssl  # Hack to get around SSL certificate of the segmentation_models_pytorch being out of date
ssl._create_default_https_context = ssl._create_unverified_context


def build_model(config, device, phases):
    from prediction.loader import ForceDataset  # hacky shit

    if config.FORCE_CLASSIFICATION:
        weight = [float(config.FORCE_CLASSIFICATION_NONZERO_WEIGHT)] * config.NUM_FORCE_CLASSES
        weight[0] = 1
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())
        out_channels = config.NUM_FORCE_CLASSES
    else:
        criterion = torch.nn.MSELoss()
        out_channels = 1

    if config.NETWORK_TYPE == 'smp':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn = resnet_preprocessor
    else:
        raise ValueError('Unknown model')

    model = model.to(device)

    out_dict = dict()
    out_dict['model'] = model
    out_dict['criterion'] = criterion

    if 'train' in phases:
        out_dict['train_dataset'] = ForceDataset(config, config.TRAIN_FILTER,
                                                 image_method=config.DATALOADER_IMAGE_METHOD,
                                                 # augmentation=get_training_augmentation(),
                                                 force_method=config.DATALOADER_FORCE_METHOD,
                                                 skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES,
                                                 preprocessing_fn=preprocessing_fn,
                                                 use_cameras=config.TRAIN_USE_CAMERAS,
                                                 phase="train",
                                                 bg_sub=config.TRAIN_BG_SUB)

    if 'val' in phases:
        # include_raw_force is true if we're not training
        out_dict['val_dataset'] = ForceDataset(config, config.VAL_FILTER, image_method=config.DATALOADER_IMAGE_METHOD,
                                                force_method=config.DATALOADER_FORCE_METHOD,
                                                skip_frames=config.DATALOADER_TEST_SKIP_FRAMES,
                                                bg_sub=config.VAL_BG_SUB,
                                                preprocessing_fn=preprocessing_fn,
                                                include_raw_force=True,
                                                use_cameras=config.VAL_USE_CAMERAS,
                                                phase="val")

    if 'test' in phases:
        raise ValueError('NO TESTING YET!!!')

    return out_dict

