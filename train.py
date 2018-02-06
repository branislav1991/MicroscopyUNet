import numpy as np
import tensorflow as tf

from models.unet.model import Model, UNetTrainConfig
#from models.pspnet.model import Model, PSPNetTrainConfig
from data_provider import DataProvider_old
from common import create_folder

# create checkpoint folder
create_folder(Model.CHECKPOINT_DIR)

# initialize model
print("Initializing model ...")
model = Model(num_scales=1)

print("Initializing data provider ...")
data_provider = DataProvider_old(model)

# load training data
print("Loading training images and masks ... ")
X_train, Y_train, sizes_train, _ = data_provider.load_train_images_resize(preprocessing=['Lab'], augmentation={'elastic_rnd': 1})
#X_train, Y_train, sizes_train, _ = load_train_images_crop(TRAIN_PATH, 
#   Model.IMG_HEIGHT, Model.IMG_WIDTH, preprocessing=['Lab'], augmentation={'elastic_rnd': 5})
print("Done loading images!")

# split training data for training and validation
X_train, Y_train, X_val, Y_val = data_provider.train_val_split(X_train, Y_train, 0.2)

# random shuffle training dataset
X_train, Y_train = data_provider.shuffle_dataset(X_train, Y_train)

print("Beginning training ... ")
model.train(X_train, Y_train, UNetTrainConfig(display_rate = 1), X_val, Y_val)
#model.train(X_train, Y_train, PSPNetTrainConfig(display_rate = 10), X_val, Y_val)
print("Done training!")