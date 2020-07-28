import os
import pandas as pd
import numpy as np

sub_0 = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/ISIC_2020/other_submissions/0.9419.csv')
sub_1 = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/ISIC_2020/other_submissions/0.9454.csv')

sub0_image = np.array(sub_0['image_name'])
sub1_image = np.array(sub_1['image_name'])

sub0_score = np.array(sub_0['target'])
sub1_score = np.array(sub_1['target'])

image_name = []
target = []

for img_name in sub1_image:
    score = np.mean([sub0_score[sub0_image == img_name], sub1_score[sub1_image == img_name]])
    image_name.append(img_name)
    target.append(score)

test_csv = pd.DataFrame({'image_name': image_name, 'target': target})
test_csv.to_csv(os.path.join('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/ISIC_2020/other_submissions', 'submission.csv'), index=False)