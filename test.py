import os

import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label, binary_opening, binary_closing, disk
import scipy

from models.unet.model import Model
from data_provider import TestDataProvider, TrainDataProviderResize
from common import mIoU, create_folder, create_predicted_folders, adaptive_threshold

# Set some parameters
SEGMENTATION_THRESHOLD = 0.5

# initialize model
print("Initializing model ...")
model = Model(num_scales=1)

print('Loading training images and masks ... ')
train_path='./data/stage1_train_small/'
#train_path='./data/stage1_train/'

train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
data_provider_train = TrainDataProviderResize(model, train_ids, preprocessing=['Lab'])
create_predicted_folders(train_ids)

print("Beginning testing on training data ...")
loss, Y_p = model.test(data_provider_train)

Y_p = np.concatenate(Y_p)
Y_p = Y_p > SEGMENTATION_THRESHOLD
#Y_p = adaptive_threshold(Y_p)

mIoU_value = mIoU(data_provider_train.get_true_Y(), Y_p)
print("Training data loss: {0}, Mean IoU: {1}".format(loss, mIoU_value))

print("Saving generated masks ...")
train_sizes = data_provider_train.get_sizes()
for n in tqdm(range(0, Y_p.shape[0]), total=Y_p.shape[0]):
    path = train_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0].astype(np.uint8) * 255, train_sizes[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)

print("Loading test images ...")
test_path='./data/stage1_test/'
test_ids = next(os.walk(test_path))
test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]

data_provider_test = TestDataProvider(model, test_ids, res=True, preprocessing=['Lab'])
create_predicted_folders(test_ids)

print("Beginning testing on test data ...")
_, Y_p = model.test(data_provider_test)
Y_p = list(map(lambda y: y > SEGMENTATION_THRESHOLD, Y_p))

print("Saving generated masks ...")
test_sizes = data_provider_test.get_sizes()
for n in tqdm(range(0, len(Y_p)), total=len(Y_p)):
    path = test_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n][0,:,:,0].astype(np.uint8) * 255, test_sizes[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)
print("Done!")