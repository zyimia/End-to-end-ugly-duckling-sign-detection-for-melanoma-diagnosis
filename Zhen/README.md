
## zhen's implementation
![](https://github.com/zyimia/SIIMISIC2020/blob/master/Zhen/configs/implementation.png=300x200)
### Step 1: remove duplicates from training data
```
# 486 images repeated in the ISIC2020 training data
# hence we need need to remove these data
run /Zhen/data_proc/remove_duplicate.py  
```
### Step 2: add external data
```
# merge multiple external skin data source
# since isic 2020 is higly imblance, ~30000 benign vs. ~580 malignant, thus we only add melanoma from other data source
run Zhen/data_proc/merge_csv.py
```

### Step 3: generate hair mask for augmentation
```
# generate hair mask array from images with hairs, please first select a group of images with hairs from the dataset
# or you can directly download from https://drive.google.com/file/d/1MNmHMP7DZMyImdrqrO0WZhiZyGB-VCIy/view?usp=sharing
run Zhen/data_proc/img_hair_syn.py
```

### Step 4: training & testing
```
run Zhen/main.py
run Zhen/inference/evaluation_aug.py
```
