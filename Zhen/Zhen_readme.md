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
