import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label, binary_opening, binary_closing, disk
import scipy

from models.unet.model import Model, UNetTestConfig
from data_provider import DataProvider
from common import mIoU, create_folder, create_predicted_folders

# Set some parameters
SEGMENTATION_THRESHOLD = 0.5

# initialize model
print("Initializing model ...")
model = Model(num_scales=2)

print("Initializing data provider ...")
data_provider = DataProvider(model)

print('Loading training and testing images and masks ... ')
X_train, Y_train, sizes_train, train_ids = data_provider.load_train_images_resize(preprocessing=['Lab'])
X_test, sizes_test, test_ids = data_provider.load_test_images(preprocessing=['Lab'])
print('Done loading images!')

create_predicted_folders(train_ids)
create_predicted_folders(test_ids)

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
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0].astype(np.uint8) * 255, sizes_train[n], interp='nearest')
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
    mask_resized = scipy.misc.imresize(Y_p[n,:,:,0].astype(np.uint8) * 255, sizes_test[n], interp='nearest')
    io.imsave(path + "/mask.tif", mask_resized)
print("Done!")