import numpy as np
from numpy import linalg as LA
import torch
import torch.utils.data as data
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from .stanford2d3d import Stanford2D3D
from .matterport3d import Matterport3D
from .metropolis import Metropolis

datasets_dict = {"stanford2d3d": Stanford2D3D,
                 "matterport3d": Matterport3D,
                 "metropolis": Metropolis}

def fetch_val_dataloaders(args):
    """ Create the data loader for the corresponding validation set """

    height = args.height
    width = args.width

    val_loaders = []
    for dataset_name in args.val_datasets:
        val_dataset = datasets_dict[dataset_name](height, width, is_training=False)
        val_loader = data.DataLoader(val_dataset, args.batch_size, False,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=False)
        val_loaders.append(val_loader)

    return val_loaders

