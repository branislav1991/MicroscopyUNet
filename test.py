import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label, binary_opening, binary_closing, disk
import scipy

from models.unet.model import Model
from data_provider import DataProvider_old, TestDataProvider, TrainDataProviderResize
from common import mIoU, create_folder, create_predicted_folders, adaptive_threshold

# Set some parameters
SEGMENTATION_THRESHOLD = 0.2

# initialize model
print("Initializing model ...")
model = Model(num_scales=1)

print("Initializing data provider ...")
data_provider = DataProvider_old(model)

print('Loading training images and masks ... ')
#X_train, Y_train, sizes_train, train_ids = data_provider.load_train_images_resize(preprocessing=['Lab'])
#X_test, sizes_test, test_ids = data_provider.load_test_images(res=False, preprocessing=['Lab'])
train_dp = TrainDataProviderResize(model, preprocessing=['Lab'])
create_predicted_folders(train_dp.get_ids())

print("Beginning testing on training data ...")
loss, Y_p = model.test(train_dp)

Y_p = list(map(lambda y: y > SEGMENTATION_THRESHOLD, Y_p))
#Y_p = adaptive_threshold(Y_p)

mIoU_value = mIoU(train_dp.get_true_Y(), np.concatenate(Y_p))
print("Training data loss: {0}, Mean IoU: {1}".format(loss, mIoU_value))

print("Saving generated masks ...")
train_ids = train_dp.get_ids()
train_sizes = train_dp.get_sizes()
for n in tqdm(range(0, len(Y_p)), total=len(Y_p)):
    path = train_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0].astype(np.uint8) * 255, train_sizes[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)

print("Loading test images ...")
test_dp = TestDataProvider(model, res=True, preprocessing=['Lab'])
create_predicted_folders(test_dp.get_ids())

print("Beginning testing on test data ...")
_, Y_p = model.test(test_dp)
Y_p = list(map(lambda y: y > SEGMENTATION_THRESHOLD, Y_p))

print("Saving generated masks ...")
test_ids = test_dp.get_ids()
test_sizes = test_dp.get_sizes()
for n in tqdm(range(0, len(Y_p)), total=len(Y_p)):
    path = test_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n][0,:,:,0].astype(np.uint8) * 255, test_sizes[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)
print("Done!")