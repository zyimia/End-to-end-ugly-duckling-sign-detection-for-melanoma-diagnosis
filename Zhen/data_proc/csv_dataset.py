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
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class CSV_Dataset(Dataset):
    def __init__(self, data_root, data_csv, is_train=True, fold=0):
        super(CSV_Dataset, self).__init__()

        self.n_class = 1
        self.fold = fold
        self.data_root = data_root
        self.is_train = is_train     # training data and validation data have different image sampling method
        self.samples = self.generate_folds(pd.read_csv(data_csv))

        if is_train:
            self.images = self.samples['train_images']
            self.labels = self.samples['train_labels']

            self.transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                                                        saturation=(0.8, 1.2),
                                                                        hue=(-0.05, 0.05)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        else:
            self.images = self.samples['val_images']
            self.labels = self.samples['val_labels']
            self.transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        assert len(self.images) == len(self.labels)

    def copy_img_for_balance(self, image, aug_rate):
        if aug_rate == 0:
            image = image
        else:
            image = list(image)
            image = np.concatenate([np.repeat(x, aug_rate) for x in image])
        return image

    def generate_folds(self, samples):
        images = np.array(samples['image_name'])
        labels = np.array(samples['benign_malignant'])
        unique_labels = np.unique(labels)
        splits = OrderedDict({'train_images': [], 'train_labels': [], 'val_images': [], 'val_labels': []})

        if self.is_train:
            aug_rate = [0, 56]
        else:
            aug_rate = [0, 0]

        for i in range(len(unique_labels)):
            index = np.where(labels == unique_labels[i])
            img = images[index]
            train_keys, val_keys = self.k_split(img)  # array
            # copy image for balance
            train_keys = self.copy_img_for_balance(train_keys, aug_rate[i])
            splits['train_images'] += list(train_keys)
            splits['train_labels'] += list(np.repeat(unique_labels[i], len(train_keys)))

            splits['val_images'] += list(val_keys)
            splits['val_labels'] += list(np.repeat(unique_labels[i], len(val_keys)))

        return splits

    def k_split(self, images):
        kfold = KFold(n_splits=5, shuffle=False)
        splits = []
        for i, (train_idx, val_idx) in enumerate(kfold.split(images)):
            train_keys = images[train_idx]
            val_keys = images[val_idx]

            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = val_keys

        return splits[self.fold]['train'], splits[self.fold]['val']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # select case
        # print(self.images[idx])
        # print(self.labels[idx])
        img_dir = os.path.join(self.data_root, self.images[idx] + '.jpg')
        # # load data
        img = PIL.Image.open(img_dir)
        img = self.transform(img)
        # print(self.labels[idx])

        if self.labels[idx] == 'benign':
            label = 0
        elif self.labels[idx] == 'malignant':
            label = 1
        else:
            raise ValueError

        label = torch.tensor(label)   # dtype = torch.int64

        return {'image': img, 'target': label}


if __name__ == '__main__':
    data_root = '/home/zyi/My_disk/ISIC_2020/data/train'

    sample_dataset = CSV_Dataset('/home/zyi/My_disk/ISIC_2020/data/train',
                                     '/home/zyi/My_disk/ISIC_2020/train.csv',
                                     is_train=True, fold=0)
    print(len(sample_dataset))
    sample_dataset[0]
    # idx = 182 # np.random.randint(0, len(case_list))
    # test_img = sample_dataset[idx]['image'].transpose(0, 1).transpose(1, 2).numpy()*255
    # cv2.normalize(test_img, test_img, 0, 255, cv2.NORM_MINMAX)
    # io.imshow(test_img.astype(np.uint8))
    # plt.show()
    # fig, axes = plt.subplots(1, seq_length-1)
    # sequence = sample_dataset[idx]['image']['MIC']
    # sequence = sequence['images']
    #
    # for i in range(seq_length-1):
    #     img = sequence[i, ...].numpy().transpose(1, 2, 0)*255
    #     print('max img: ', np.max(img), 'min img: ', np.min(img))
    #     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #     axes[i].imshow(img.astype(np.uint8))
    #
    # # img = PIL.Image.fromarray(sequence[0, ...].transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
    # # img.show()
    # # print(train_list[idx])
    # plt.show()
