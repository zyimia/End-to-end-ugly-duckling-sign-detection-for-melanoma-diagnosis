## Usage

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
run Zhen/data_proc/img_hair_syn.py
```

### Step 4: training & testing
```
run Zhen/main.py
run Zhen/inference/evaluation_aug.py
```
