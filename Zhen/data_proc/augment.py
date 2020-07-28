import numpy as np
import torch
import random
import PIL
import cv2
import os
import skimage
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from _collections import OrderedDict
from data_proc.inception_crop import InceptionCrop
import albumentations as A
# https://github.com/fabioperez/skin-data-augmentation/blob/master/train.py

aug_parameters = OrderedDict({'affine': {'rotation': 90,
                                         'shear': 20,
                                         'scale': [0.8, 1.2]},
                              'hflip': True,
                              'vflip': True,
                              'rotate': True,
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


class Random_rotate:
    def __init__(self, p=0.5):
        self.transform = A.Rotate(p=p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class Realistic_hair_aug:
    def __init__(self, hair_library=None, p=0.5):
        assert os.path.exists(hair_library)
        self.hair_array = np.load(hair_library)
        self.ratio = p

    def add_hair(self, image):
        image = np.array(image)

        mask = self.hair_array[np.random.randint(self.hair_array.shape[0]), :, :]
        image_resize = cv2.resize(image, mask.shape[:2])

        image_hair = cv2.bitwise_and(image_resize, image_resize, mask=mask)
        image_hair = skimage.filters.gaussian(image_hair, sigma=1)
        image_hair = cv2.resize(image_hair, image.shape[:2])
        image_hair = image_hair*255

        return image_hair.astype(np.uint8)

    def __call__(self, image):
        if random.random() > self.ratio:
            try:
                image = self.add_hair(image)
                image = PIL.Image.fromarray(image)
            except TypeError:
                image = image

        return image


class Affine:
    def __init__(self, augs):
        self.affine = iaa.Affine(
            rotate=(-augs['rotation'], augs['rotation']),
            shear=(-augs['shear'], augs['shear']),
            scale=({'x': augs['scale'][0], 'y': augs['scale'][1]}),
            mode='symmetric')

    def random_affine(self, image):
        images = np.array(image)
        assert images.shape[-1] % 3 == 0
        images = self.affine.augment_image(images)

        return PIL.Image.fromarray(images)

    def __call__(self, image):

        if random.random() < 0.5:
            image = self.random_affine(image)

        return image


class Augmentations:
    def __init__(self, augs, tta='ten'):
        transform_list = []
        self.mean = augs['normalization']['mean']
        self.std = augs['normalization']['std']
        self.size = augs['size']
        # random resize and crop
        transform_list.append(transforms.RandomResizedCrop(size=augs['size'],
                                                           scale=augs['scale'],
                                                           ratio=augs['ratio'])
                              )
        # random affine
        if augs['affine'] is not None:
            transform_list.append(Affine(augs['affine']))

        # random flip
        if augs['hflip'] is not None:
            transform_list.append(transforms.RandomHorizontalFlip())

        if augs['vflip'] is not None:
            transform_list.append(transforms.RandomVerticalFlip())

        # random color trans
        if augs['color_trans'] is not None:
            transform_list.append(transforms.ColorJitter(augs['color_trans']['brightness'],
                                                         augs['color_trans']['contrast'],
                                                         augs['color_trans']['saturation'],
                                                         augs['color_trans']['hue'])
                                  )
        if augs['rotate'] is not None:
            transform_list.append(Random_rotate())

        if augs['hair_mask'] is not None:
            transform_list.append(Realistic_hair_aug(hair_library=augs['hair_mask']))

        # normalize to tensor
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.transform = transforms.Compose(transform_list)  # output: T, H, W, C
        self.ta_transform = self._get_crop_transform(tta)

    def _get_crop_transform(self, method='ten'):

        if method == 'ten':
            crop_tf = transforms.Compose([
                transforms.Resize((self.size + 32, self.size + 32)),
                transforms.TenCrop((self.size, self.size))
            ])

        if method == 'inception':
            crop_tf = InceptionCrop(
                self.size,
                resizes=tuple(range(self.size + 32, self.size + 129, 32))
            )

        after_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return transforms.Compose([
            crop_tf,
            transforms.Lambda(
                lambda crops: torch.stack(
                    [after_crop(crop) for crop in crops]))
        ])


