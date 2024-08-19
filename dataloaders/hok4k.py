import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices,image_loader
import pandas as pd

class HOK4K(Dataset):
    def __init__(self, split, data_root, num_labels, max_samples=-1,transform=None,testing=False,known_labels=0):
        self.split = split
        self.data_root = data_root
        self.data = pd.read_csv(os.path.join(data_root, 'car_imgs_4000.csv'))
        self.num_images = self.data.shape[0]
        self.transform = transform
        self.num_labels = num_labels
        self.testing = testing
        self.known_labels = known_labels
        if self.split == 'train':
            self.data_list = range(0, int(self.num_images*0.8))
        else:
            self.data_list = range(int(self.num_images*0.8), self.num_images)

        if max_samples != -1:
            self.data_list = self.data_list[0:max_samples]
        self.epoch = 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, clamp_labels=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.data_root, 'imgs', self.data.iloc[idx,0])
        image = image_loader(img_path, self.transform)
        if clamp_labels:
            labels = [0 if self.data.iloc[idx,i] ==0 else 1 for i in range(1,self.data.shape[1])]
        else:
            labels = [self.data.iloc[idx,i] for i in range(1,self.data.shape[1])]
        labels = torch.Tensor(labels)
        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = self.data.iloc[idx,0]
        return sample

class HOK4KVis(HOK4K):
    def __init__(self, split, data_root, num_labels, max_samples=-1,transform=None,testing=False,known_labels=0):
        super().__init__(split, data_root, num_labels, max_samples,transform,testing,known_labels)

    def __getitem__(self, idx):
        return super().__getitem__(idx, clamp_labels=False)