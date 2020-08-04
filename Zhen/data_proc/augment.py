import numpy as np
import torch
import random
import PIL
import cv2
import os
import skimage
import skimage.io as io
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from _collections import OrderedDict
from data_proc.inception_crop import InceptionCrop
import albumentations as A
# https://github.com/fabioperez/skin-data-augmentation/blob/master/train.py


class Microscope:
    def __init__(self, p=(0.15, 0.45)):
        self.p = p

    def add_black_microscope(self, image):
        image = np.array(image)
        circle = cv2.circle((np.ones(image.shape) * 255).astype(np.uint8), (image.shape[0] // 2, image.shape[1] // 2),
                            random.randint(image.shape[0] // 2 + 0, image.shape[0] // 2 + 50), (0, 0, 0), -1)
        mask = circle - 255
        mask = skimage.filters.gaussian(mask * 255, sigma=random.randint(5, 20))
        image = np.multiply(image, mask)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        return image

    def add_white_microscope(self, image):
        image = cca.grey_world(from_pil(image))  # array np.uint8
        circle = cv2.circle((np.ones(image.shape) * 255).astype(np.uint8), (image.shape[0] // 2, image.shape[1] // 2),
                            random.randint(image.shape[0] // 2 - 10, image.shape[0] // 2 + 10), (0, 0, 0), -1)
        mask = circle - 255
        image[mask == 0] = 255
        image = skimage.filters.gaussian(image, sigma=random.random())
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        return image

    def __call__(self, image):
        ratio = random.random()
        if ratio < self.p[0]:
            image = self.add_white_microscope(image)

        if self.p[0] < ratio < self.p[1]:
            image = self.add_black_microscope(image)

        return image


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
        # image_hair = image_hair*255
        image_hair = cv2.normalize(image_hair, None, 0, 255, cv2.NORM_MINMAX)

        return image_hair.astype(np.uint8)

    def __call__(self, image):
        if random.random() < self.ratio:
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

        if augs['microscope'] is not None:
            transform_list.append(Microscope(p=augs['microscope']))

        # normalize to tensor
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.debug_trans = transforms.Compose(transform_list[:-2])
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


if __name__ == '__main__':
    aug_parameters = OrderedDict({'affine': None,
                                  'hflip': True,
                                  'vflip': True,
                                  'rotate': True,
                                  'hair_mask': '/home/zyi/MedicalAI/ISIC_2020/hairs2/mask_array.npy',
                                  'microscope': (0.1, 0.4),
                                  'color_trans': {'brightness': (0.7, 1.3),
                                                  'contrast': (0.7, 1.3),
                                                  'saturation': (0.7, 1.3),
                                                  'hue': (-0.1, 0.1)},
                                  'normalization': {'mean': (0.485, 0.456, 0.406),
                                                    'std': (0.229, 0.224, 0.225)},
                                  'size': 320,
                                  'scale': (0.7, 1.3),
                                  'ratio': (0.7, 1.3)
                                  }
                                 )

    augmentor = Augmentations(augs=aug_parameters)

    # img = io.imread('/home/zyi/MedicalAI/ISIC_2020/data/test/ISIC_0165615.jpg')
    # img = cv2.resize(img, (img.shape[1]//6, img.shape[0]//6))
    # # img = cv2.resize(img, (320, 320))
    # circle = cv2.circle((np.ones([320, 320, 3]) * 255).astype(np.uint8), (320 // 2, 320 // 2),
    #                     random.randint(320 // 2 + 15, 320 // 2 + 30), (0, 0, 0), -1)
    #
    # im1 = cca.grey_world(img)
    # print(im1.dtype)
    # # mask = circle - 255
    # circle = cv2.resize(circle, (img.shape[0], img.shape[1])).transpose(1,0,2)
    # im1[circle == 255] = 0
    # image = PIL.Image.fromarray(im1)
    # image.show()
    image = PIL.Image.open('/home/zyi/MedicalAI/ISIC_2020/data/test/ISIC_1336046.jpg')
    img = augmentor.debug_trans(image)
    plt.imshow(np.array(img))
    plt.show()
