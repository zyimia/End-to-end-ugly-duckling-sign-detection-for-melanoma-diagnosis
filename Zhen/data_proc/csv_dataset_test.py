import os
import cv2
import PIL
import torch
import pandas as pd
import skimage.io as io
import numpy as np
from glob import glob
import pickle as pkl
from sklearn.model_selection import KFold
from collections import OrderedDict
from torch.utils.data import Dataset
from data_proc.augment import Augmentations
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class SIIM_Dataset(Dataset):
    def __init__(self, data_root, data_csv, case_list=None, transform=None):
        super(SIIM_Dataset, self).__init__()

        self.n_class = 1
        self.case_list = case_list
        self.data_root = data_root
        self.transform = transform
        self.samples = pd.read_csv(data_csv)

        self.images = np.array(self.samples['image_name'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # select case
        img_dir = os.path.join(self.data_root, self.images[idx] + '.jpg')
        # load data
        img = PIL.Image.open(img_dir)
        img = self.transform(img)

        return {'image': img, 'name': self.images[idx]}


if __name__ == '__main__':
    aug_parameters = OrderedDict({'affine': {'rotation': 90,
                                             'shear': 20,
                                             'scale': [0.8, 1.2]},
                                  'hflip': True,
                                  'vflip': True,
                                  'color_trans': {'brightness': (0.7, 1.3),
                                                  'contrast': (0.7, 1.3),
                                                  'saturation': (0.7, 1.3),
                                                  'hue': (-0.1, 0.1)},
                                  'normalization': {'mean': (0.485, 0.456, 0.406),
                                                    'std': (0.229, 0.224, 0.225)},
                                  'size': 320,
                                  'scale': (0.8, 1.2),
                                  'ratio': (0.8, 1.2)
                                  }
                                 )

    data_root = '/home/zyi/My_disk/ISIC_2020/data/test'

    augmentor = Augmentations(augs=aug_parameters, tta='ten')
    sample_dataset = SIIM_Dataset('/home/zyi/My_disk/ISIC_2020/data/test',
                                  '/home/zyi/My_disk/ISIC_2020/test.csv',
                                  transform=augmentor.ta_transform)
    print(len(sample_dataset))
    print(sample_dataset[0]['name'])
    print(sample_dataset[0]['image'].shape)
    idx = 182  # np.random.randint(0, len(case_list))
    test_img = sample_dataset[idx]['image'][0].transpose(0, 1).transpose(1, 2).numpy()
    print(test_img.shape)
    out_img = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX)
    io.imshow(out_img.astype(np.uint8))
    plt.show()