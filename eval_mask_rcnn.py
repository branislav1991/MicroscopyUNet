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
from tqdm import tqdm
from skimage import io
import json

from models.mask_rcnn.cell_dataset import CellsDataset
from models.mask_rcnn.config import CellConfig
from models.mask_rcnn import utils
from models.mask_rcnn import model as modellib
from models.mask_rcnn.model import log

# TODO: Remove this and make a nicer file structure
from models.unet.common import create_predicted_folders

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to load checkpoints from
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "mask_rcnn")
TENSORBOARD_DIR = os.path.join(ROOT_DIR, ".tensorboard", "mask_rcnn")

MAP_FNAME = "evals.json"

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def eval_mAP(test_ids, test_path, checkpoint_dir):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    inference_config = InferenceConfig()

    # Create the model in inference mode
    print("Initializing model in inference mode ... ")
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            checkpoint_dir=checkpoint_dir,
                            tensorboard_dir=TENSORBOARD_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    print("Loading trained weights ... ")
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    print('Loading training images ... ')
    create_predicted_folders(test_ids)

    dataset_val = CellsDataset()
    dataset_val.load_cells(test_ids)
    dataset_val.prepare()

    print('Evaluating mAP ... ')
    APs = []
    eval_json = []

    for i,image_id in enumerate(test_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                i, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        eval_json.append({"img": dataset_val.image_info[i]["simple_path"], "AP": AP})
        APs.append(AP)

    with open(os.path.join(test_path, MAP_FNAME), 'w') as fp:
        json.dump(eval_json, fp)
        
    print("mAP: ", np.mean(APs))

if __name__ == "__main__":
    val_path='./data/stage1_val/'
    val_ids = next(os.walk(val_path))
    val_ids = [[val_ids[0] + d,d] for d in val_ids[1]]
    eval_mAP(val_ids, val_path, CHECKPOINT_DIR)
    