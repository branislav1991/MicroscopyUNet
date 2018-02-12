import os
import random
import math
import numpy as np
import tensorflow as tf

from models.unet.model import Model, UNetTrainConfig
#from models.pspnet.model import Model, PSPNetTrainConfig
from data_provider import TrainDataProviderMulticlass, TrainDataProviderTilingMulticlass
from common import create_folder

RESTORE = False
VALIDATION_FRACTION = 0.2
TILE_HEIGHT = 280
TILE_WIDTH = 280
NUM_CHANNELS = 1

# create checkpoint folder
create_folder(Model.CHECKPOINT_DIR)

# initialize model
print("Initializing model ...")
model = Model(NUM_CHANNELS)

print('Loading training images and masks ... ')
#train_path='./data/stage1_train_small/'
train_path='./data/stage1_train/'

train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

# shuffle ids randomly and separate into training and validation
random.shuffle(train_ids)
val_part = math.floor(len(train_ids) * VALIDATION_FRACTION)
val_ids = train_ids[:val_part]
train_ids = train_ids[val_part:]

data_provider_train = TrainDataProviderTilingMulticlass(TILE_HEIGHT, TILE_WIDTH, NUM_CHANNELS, train_ids, batch_size=4, shuffle=True, preprocessing=['Lab'], num_tiles=50)
# for the validation data provider batch size must be 1 since we are using a fully convolutional
# network inference with varying input sizes!
data_provider_val = TrainDataProviderMulticlass(model, val_ids, preprocessing=['Lab'])

print("Beginning training ... ")
model.train(UNetTrainConfig(val_rate = 1), data_provider_train, restore=RESTORE, data_provider_val=data_provider_val)
print("Done training!")