import numpy as np
import tensorflow as tf

from models.unet.model import Model, UNetTrainConfig
#from models.mask_rcnn.model import Model
from common import create_folder, load_train_images

# create checkpoint folder
create_folder(Model.CHECKPOINT_DIR)

#TRAIN_PATH = './data/stage1_train/'
TRAIN_PATH = './data/stage1_train_small/'

# load training data
print("Getting and resizing train images and masks ... ")
X_train, Y_train, sizes_train, _ = load_train_images(TRAIN_PATH, Model.IMG_HEIGHT, Model.IMG_WIDTH, Model.IMG_CHANNELS)
print("Done loading images!")

# initialize model
print("Initializing model ...")
model = Model()

print("Beginning training ... ")
model.train(X_train, Y_train, UNetTrainConfig())
print("Done training!")