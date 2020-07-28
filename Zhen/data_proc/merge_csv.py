import os
import numpy as np
import pandas as pd
from glob import glob


# ISIC 2020 data
ISIC_2020_data_root = '/home/zyi/MedicalAI/ISIC_2020/data/train'
ISIC_2020_csv = pd.read_csv('/home/zyi/My_disk/ISIC_2020/train.csv')
ISIC_2020_image = np.array(ISIC_2020_csv['image_name'])
ISIC_2020_label = np.array(ISIC_2020_csv['benign_malignant'])

ISIC_2020_image = [os.path.join(ISIC_2020_data_root, x + '.jpg') for x in ISIC_2020_image]
ISIC_2020_label = [label for label in ISIC_2020_label]
assert len(ISIC_2020_label) == len(ISIC_2020_image)

# ISIC 2019 data
ISIC_2019_data_root = '/home/zyi/My_disk/ISIC_2020/ext_data/ISIC_2019_Training'
csv = pd.read_csv('/home/zyi/My_disk/ISIC_2020/ext_data/ISIC_2019_Training_GroundTruth.csv')
ISIC_2019_image = np.array(csv['image'])
ISIC_2019_label = np.array(csv['MEL'])

ISIC_2019_ext_image = ISIC_2019_image[ISIC_2019_label == 1]
ISIC_2019_image = [os.path.join(ISIC_2019_data_root, x +'.jpg') for x in ISIC_2019_ext_image]
ISIC_2019_label = ['malignant' for x in range(len(ISIC_2019_ext_image))]
assert len(ISIC_2019_image) == len(ISIC_2019_label)

# HAM10000
HAM_10000_data_root = '/home/zyi/My_disk/ISIC_2020/ext_data/HM_10000/images'
HAM_10000_csv = pd.read_csv('/home/zyi/My_disk/ISIC_2020/ext_data/HM_10000/HAM10000_metadata.csv')
HAM_10000_image = np.array(HAM_10000_csv['image_id'])
HAM_10000_label = np.array(HAM_10000_csv['dx'])

HAM_10000_ext_image = HAM_10000_image[HAM_10000_label == 'mel']
HAM_10000_image = [os.path.join(HAM_10000_data_root, x + '.jpg') for x in HAM_10000_ext_image]
HAM_10000_label = ['malignant' for x in range(len(HAM_10000_image))]
assert len(HAM_10000_label) == len(HAM_10000_image)


# molemap data
molemap_data_root = '/home/zyi/My_disk/ISIC_2020/ext_data/molemap'
molemap_image = glob(os.path.join(molemap_data_root, '*MIC*'))
molemap_label = ['malignant' for x in range(len(molemap_image))]

new_dataset_image = ISIC_2019_image + HAM_10000_image + molemap_image  #+ ISIC_2020_image
new_dataset_label = ISIC_2019_label + HAM_10000_label + molemap_label  # ISIC_2020_label

new_csv = pd.DataFrame({'image_name': new_dataset_image, 'target': new_dataset_label})
print(len(new_csv))
new_csv.to_csv(os.path.join('/home/zyi/My_disk/ISIC_2020', 'ext_dataset.csv'), index=False)


csv = pd.read_csv('/home/zyi/MedicalAI/ISIC_2020/ext_dataset.csv')
img_name = np.array(csv['image_name'])
target = np.array(csv['target'])

for i in range(len(img_name)):
    img_name[i] = img_name[i].replace('/home/zyi/My_disk/ISIC_2020/', '/home/zyi/MedicalAI/ISIC_2020/')

new_csv = pd.DataFrame({'image_name': img_name, 'target': target})
print(len(new_csv))
new_csv.to_csv(os.path.join('/home/zyi/MedicalAI/ISIC_2020', 'ext_dataset1.csv'), index=False)
