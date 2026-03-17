from __future__ import print_function
import os
import cv2
import numpy as np
import random
from glob import glob

import torch
from torch.utils import data
from torchvision import transforms


def sort_key(s):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


class Real(data.Dataset):
    """The Real Dataset"""

    def __init__(self, height=518, width=1036, **kwargs):
        """
        Args:
            height, width: input size.
        """
        self.v = 1
        self.w = width
        self.h = height
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __mul__(self, v):
        self.v = v
        return self

    def __len__(self):
        return len(self.rgb_list)*self.v

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            idx = [i % (len(self.rgb_list) * self.v) for i in idx]
            idx = [i % len(self.rgb_list) for i in idx]
        else:
            idx = idx % (len(self.rgb_list) * self.v)
            idx = idx % len(self.rgb_list)

        inputs = {}

        rgb_name = os.path.join(self.rgb_list[idx])
        rgb = cv2.imread(rgb_name)
            
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        return inputs



