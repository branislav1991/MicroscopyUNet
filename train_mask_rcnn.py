import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from models.mask_rcnn.config import CellConfig
from models.mask_rcnn import utils
from models.mask_rcnn import model as modellib
from models.mask_rcnn.model import log
from models.mask_rcnn.cell_dataset import CellsDataset

def create_folder(f):
    if not os.path.exists(f):
        os.makedirs(f)

VALIDATION_FRACTION = 0.2
LEARNING_RATE = 0.001

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "mask_rcnn")
create_folder(CHECKPOINT_DIR)
TENSORBOARD_DIR = os.path.join(ROOT_DIR, ".tensorboard", "mask_rcnn")
create_folder(TENSORBOARD_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def train(train_path, val_path, init_with, procedures, config=None):
    print('Loading training images and masks ... ')

    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

    val_ids = next(os.walk(val_path))
    val_ids = [[val_ids[0] + d,d] for d in val_ids[1]]

    # shuffle ids randomly and separate into training and validation
    # random.shuffle(train_ids)
    # val_part = math.floor(len(train_ids) * VALIDATION_FRACTION)
    # val_ids = train_ids[:val_part]
    # train_ids = train_ids[val_part:]

    dataset_train = CellsDataset()
    dataset_train.load_cells(train_ids)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellsDataset()
    dataset_val.load_cells(val_ids)
    dataset_val.prepare()

    if config is None:
        config = CellConfig()

    # Create model in training mode
    print('Initializing model ... ')
    model = modellib.MaskRCNN(mode="training", config=config, checkpoint_dir=CHECKPOINT_DIR,
                            tensorboard_dir=TENSORBOARD_DIR)

    # Which weights to start with?
    print('Loading weights ... ')

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    print("Beginning training ... ")
    histories = []
    for p in procedures:
        history = model.train(dataset_train, dataset_val, 
                learning_rate=p["learning_rate"],
                epochs=p["epochs"], 
                layers=p["layers"])
        histories.append(history)

    print("Done training!")
    return histories

if __name__ == "__main__":
    #train_path='./data/stage1_train_small/'
    train(train_path=".\\data\\stage1_train\\", val_path=".\\data\\stage1_val\\", init_with="coco", 
          procedures=[{"layers": "heads", "learning_rate": LEARNING_RATE, "epochs": 2}, 
                      {"layers": "5+", "learning_rate": LEARNING_RATE/2, "epochs": 10},
                      {"layers": "3+", "learning_rate": LEARNING_RATE/5, "epochs": 20}
                      {"layers": "all", "learning_rate": LEARNING_RATE/10, "epochs": 10}])