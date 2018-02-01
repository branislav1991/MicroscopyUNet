import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label, binary_opening, binary_closing, disk
import scipy

from models.unet.model import Model, UNetTestConfig
from common import mIoU, create_folder, create_predicted_folders, load_train_images, load_test_images

# Set some parameters
#TRAIN_PATH = './data/stage1_train/'
TRAIN_PATH = './data/stage1_train_small/'
TEST_PATH = './data/stage1_test/'
SEGMENTATION_THRESHOLD = 0.9

print('Getting and resizing train images and masks ... ')
X_train, Y_train, sizes_train, train_ids = load_train_images(TRAIN_PATH, 
    Model.IMG_HEIGHT, Model.IMG_WIDTH, preprocessing=['Lab'])
X_test, sizes_test, test_ids = load_test_images(TEST_PATH, Model.IMG_HEIGHT,
    Model.IMG_WIDTH, preprocessing=['Lab'])
print('Done loading images!')

create_predicted_folders(train_ids)
create_predicted_folders(test_ids)

# initialize model
print("Initializing model ...")
model = Model()

print("Beginning testing on training data ...")
loss, Y_p = model.test(X_train, UNetTestConfig(), Y_train)

Y_p = Y_p > SEGMENTATION_THRESHOLD
# for i in range(0,Y_p.shape[0]):
#     Y_p[i,:,:,0] = binary_opening(np.squeeze(Y_p[i,:,:,0]), disk(2))
#     Y_p[i,:,:,0] = binary_closing(np.squeeze(Y_p[i,:,:,0]), disk(2))
# Y_p = Y_p.astype(bool)

mIoU_value = mIoU(Y_train, Y_p)
print("Training data loss: {0}, Mean IoU: {1}".format(loss, mIoU_value))

print("Saving generated masks ...")
for n in tqdm(range(0, Y_p.shape[0]), total=Y_p.shape[0]):
    path = train_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0], sizes_train[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)

print("Beginning testing on test data ...")
loss, Y_p = model.test(X_test, UNetTestConfig())
Y_p = Y_p > SEGMENTATION_THRESHOLD
# for i in range(0,Y_p.shape[0]):
#     Y_p[i,:,:,0] = binary_opening(np.squeeze(Y_p[i,:,:,0]), disk(2))
#     Y_p[i,:,:,0] = binary_closing(np.squeeze(Y_p[i,:,:,0]), disk(2))
Y_p = (Y_p).astype(int)

print("Saving generated masks ...")
for n in tqdm(range(0, Y_p.shape[0]), total=Y_p.shape[0]):
    path = test_ids[n][0] + "/masks_predicted"
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0], sizes_test[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)
print("Done!")