import os
import cv2
import PIL
import torch
import pandas as pd
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import OrderedDict
from torch.utils.data import Dataset
from data_proc.augment import Augmentations


class CSV_Ext_Dataset(Dataset):
    def __init__(self, data_csv, data_root, ext_csv=None, is_train=True, transform=None, fold=0):
        super(CSV_Ext_Dataset, self).__init__()

        self.n_class = 1
        self.fold = fold
        self.is_train = is_train
        self.transform = transform
        self.data_root = data_root
        self.samples = self.generate_folds(pd.read_csv(data_csv))

        if is_train:
            self.images = np.array(self.samples['train_images'])  # 26033: 467
            self.labels = np.array(self.samples['train_labels'])

            if ext_csv is not None:
                print('we are using external data')
                ext_images, ext_labels = self.add_ext_images(ext_csv)
                self.images = np.concatenate([self.images, ext_images])
                self.labels = np.concatenate([self.labels, ext_labels])  # 26033: 6447

                self.images, self.labels = self.copy_img_for_balance(self.images, self.labels)

            # print(np.unique(self.labels))
        else:
            self.images = self.samples['val_images']
            self.labels = self.samples['val_labels']

        assert len(self.images) == len(self.labels)

    def add_ext_images(self, ext_csv):
        ext_images = np.array(pd.read_csv(ext_csv)['image_name'])
        ext_labels = np.array(pd.read_csv(ext_csv)['target'])
        return ext_images, ext_labels

    def copy_img_for_balance(self, images, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 2
        assert len(images) == len(labels)

        new_images = []
        new_labels = []
        if counts[0] > counts[1]:
            new_images.append(images[labels == unique_labels[0]])
            new_labels.append(labels[labels == unique_labels[0]])

            image = np.concatenate([np.repeat(x, counts[0]//counts[1]) for x in images[labels == unique_labels[1]]])
            label = np.repeat(unique_labels[1], len(image))

            new_images.append(image)
            new_labels.append(label)

        return np.concatenate(new_images), np.concatenate(new_labels)

    def generate_folds(self, samples):
        image_list = [os.path.join(self.data_root, x + '.jpg')
                      for x in np.array(samples['image_name'])]
        images = np.array(image_list)
        labels = np.array(samples['benign_malignant'])
        unique_labels = np.unique(labels)
        splits = OrderedDict({'train_images': [], 'train_labels': [], 'val_images': [], 'val_labels': []})

        for i in range(len(unique_labels)):
            index = np.where(labels == unique_labels[i])
            img = images[index]
            train_keys, val_keys = self.k_split(img)  # array

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
        # # load data

        img = PIL.Image.open(self.images[idx])

        img = self.transform(img)

        if self.labels[idx] == 'benign':
            label = 0
        elif self.labels[idx] == 'malignant':
            label = 1
        else:
            raise ValueError

        label = torch.tensor(label)   # dtype = torch.int64
        return {'image': img, 'target': label}


if __name__ == '__main__':
    aug_parameters = OrderedDict({'affine': {'rotation': 90,
                                             'shear': 10,
                                             'scale': [0.9, 1.1]},
                                  'hflip': True,
                                  'vflip': True,
                                  'hair_mask': '/home/zyi/MedicalAI/ISIC_2020/hairs2/mask_array.npy',
                                  'rotate': True,
                                  'color_trans': {'brightness': (0.75, 1.25),
                                                  'contrast': (0.75, 1.25),
                                                  'saturation': (0.75, 1.25),
                                                  'hue': (-0.05, 0.05)},
                                  'normalization': {'mean': (0.485, 0.456, 0.406),
                                                    'std': (0.229, 0.224, 0.225)},
                                  'size': 320,
                                  'scale': (0.7, 1.3),
                                  'ratio': (0.7, 1.3)
                                  }
                                 )

    data_root = '/home/zyi/My_disk/ISIC_2020/data/train'
    augmentor = Augmentations(augs=aug_parameters, tta='ten')
    sample_dataset = CSV_Ext_Dataset('/home/zyi/MedicalAI/ISIC_2020/train.csv',
                                     data_root='/home/zyi/MedicalAI/ISIC_2020/data/train',
                                     ext_csv='/home/zyi/My_disk/ISIC_2020/ext_dataset.csv',
                                     is_train=True,
                                     transform=augmentor.transform, fold=0)
    print(len(sample_dataset))
    idx = np.random.randint(0, len(sample_dataset))
    test_img = sample_dataset[idx]['image'].transpose(0, 1).transpose(1, 2).numpy()
    test_img = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX)
    io.imshow(test_img.astype(np.uint8))
    plt.show()
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

