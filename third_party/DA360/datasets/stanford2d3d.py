from __future__ import print_function
import os
import cv2
import numpy as np
import random
from glob import glob

import torch
from torch.utils import data
from torchvision import transforms

class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""

    def __init__(self, height=518, width=1036, disable_color_augmentation=False, disable_LR_filp_augmentation=False,
                 disable_yaw_rotation_augmentation=False, is_training=False, **kwargs):
        """
        Args:
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.v = 1
        self.root_dir = "./data/Stanford2D3D/"
        self.is_training = is_training
        self.read_list()
        self.w = width
        self.h = height

        self.max_depth_meters = 10.0
        self.min_depth_meters = 0.01

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __mul__(self, v):
        self.v = v
        return self

    def __len__(self):
        return len(self.rgb_depth_list[0])*self.v

    def read_list(self):

        if self.is_training:
            scenes = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
        else:
            scenes = ['area_5a', 'area_5b']

        rgb_lists = []
        depth_lists = []
        for scene in scenes:
            rgb_list = sorted(glob(os.path.join(self.root_dir, scene, 'rgb_/*_domain_rgb.png')))
            depth_list = sorted(glob(os.path.join(self.root_dir, scene, 'depth/*_domain_depth.png')))
            assert len(rgb_list) == len(depth_list), "rgb files are inconsistent with depth files"

            rgb_lists += rgb_list
            depth_lists += depth_list

        self.rgb_depth_list = [rgb_lists, depth_lists]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            idx = [i % (len(self.rgb_depth_list[0]) * self.v) for i in idx]
            idx = [i % len(self.rgb_depth_list[0]) for i in idx]
        else:
            idx = idx % (len(self.rgb_depth_list[0]) * self.v)
            idx = idx % len(self.rgb_depth_list[0])

        inputs = {}

        rgb_name = os.path.join(self.rgb_depth_list[0][idx])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.rgb_depth_list[1][idx])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float32)/512
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > self.min_depth_meters) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        
        inputs["gt_disp"] = -torch.ones_like(inputs["gt_depth"])
        inputs["gt_disp"][inputs["val_mask"]] = 1/inputs["gt_depth"][inputs["val_mask"]]
        
        return inputs



