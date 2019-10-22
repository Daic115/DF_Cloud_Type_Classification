#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   dataset.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:36   Daic       1.0
'''
import os
import cv2
import json
import torch
import numpy as np
import albumentations as albu
from torch.utils.data import  DataLoader
from torch.utils.data import  Dataset as BaseDB


def SkyCrop(img):
    #img [h,w,3]
    if img.shape[1]*1.8<img.shape[0]:
        #too high
        fix_len = round(img.shape[1]*1.4)
        return img[:fix_len,::,::].astype(int)
    else:
        return img

def get_training_augmentation():
    #just for train
    train_transform = [
        albu.OneOf(
            [
                albu.NoOp(p=1),
                albu.RandomResizedCrop(512, 512, scale=(0.5, 1.0), ratio=(0.8, 1.2), p=1.0),
                albu.Rotate(limit=(-20, 20), p=1.),
            ],
            p=0.8,
        ),
        albu.HorizontalFlip(p=0.5),
    ]
    return albu.Compose(train_transform)

def get_preprocess(size=224):
    res=[
        albu.Resize(size, size),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return albu.Compose(res)
def get_flip_preprocess(size=224):
    res=[
        albu.HorizontalFlip(p=1.),
        albu.Resize(size, size),
        albu.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return albu.Compose(res)

class ClsSteelDataset(BaseDB):
    def __init__(self,opt, phase ,augmentation=None):
        self.phase = phase
        self.size = opt.size
        self.precrop = opt.precrop

        self.data_type = opt.data_type
        print("data type : %s"%self.data_type)

        self.image_path = opt.image_path

        self.infos = json.load(open(opt.input_json))

        if self.data_type == 'kfolder':
            self.dataset = [tmp for tmp in self.infos if tmp['split'] == self.phase]

        elif self.data_type == 'all':
            self.dataset = self.infos

        else:
            raise Exception("Unsupported data type {}".format(self.data_type))


        self.augmentation = augmentation

        self.preprocess = get_preprocess(size=self.size)

    def __getitem__(self, idx):
        name = self.dataset[idx]['img']
        img = cv2.imread(os.path.join(self.image_path, self.dataset[idx]['img']))

        if self.precrop:
            img = SkyCrop(img).astype(np.uint8)
        else:
            img = img.astype(np.uint8)


        if self.phase == 'test':
            label = -1
        else:
            label = self.dataset[idx]['cate']

        if self.phase == 'train':
            sample = self.augmentation(image=img)
            img = sample['image']

        img = self.preprocess(image=img)['image']# size size 3
        img = torch.from_numpy(img).permute(2,0,1).float()#

        return img , label , name#, mask

    def __len__(self):
        return len(self.dataset)
