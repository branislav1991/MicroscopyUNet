import os
import sys
import numpy as np
from skimage import io
from skimage.morphology import binary_closing, disk

# code for preprocessing ground truth masks
def create_processed_folders(ids):
    for pathar in ids:
        if not os.path.exists(pathar[0] + "/masks_processed"):
            os.makedirs(pathar[0] + "/masks_processed")

if len(sys.argv) < 2:
    train_path="./data/stage1_train/"
else:
    train_path = sys.argv[1]

train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

create_processed_folders(train_ids)

for path in train_ids:
    for mask_file in next(os.walk(path[0] + '/masks/'))[2]:
        mask_ = io.imread(path[0] + '/masks/' + mask_file)
        mask_ = binary_closing(mask_, disk(2))
        mask_ = mask_.astype('uint8') * 255
        io.imsave(path[0] + '/masks_processed/' + mask_file, mask_)