import os
import numpy as np
import pandas as pd
import shutil
from glob import glob

# https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg?select=duplicates_11062020.csv
# 486 image repeat

data_root = '/home/zyi/MedicalAI/ISIC_2020/data/train'
res_root = '/home/zyi/MedicalAI/ISIC_2020/data/repeat'
img_list = np.array(pd.read_csv('/home/zyi/MedicalAI/ISIC_2020/data/duplicates.csv')['image_id0'])
print(len(img_list))

for id in img_list:
    img = os.path.join(data_root, id + '.jpg')
    if os.path.exists(img):
        shutil.move(img, img.replace(os.path.dirname(img), res_root))

# re-organize train csv
csv = pd.read_csv('/home/zyi/MedicalAI/ISIC_2020/train.csv')
img_list = glob(os.path.join(res_root, '*jpg'))

for img in img_list:
    idx = os.path.basename(img).split('.')[0]
    csv = csv[csv.image_name != idx]

csv.to_csv(os.path.join('/home/zyi/MedicalAI/ISIC_2020', 'train.csv'), index=False)