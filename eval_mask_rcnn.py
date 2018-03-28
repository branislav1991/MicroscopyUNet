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

# TODO: Remove this and make a nicer file structure
from models.unet.common import create_predicted_folders

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to load checkpoints from
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "mask_rcnn")
TENSORBOARD_DIR = os.path.join(ROOT_DIR, ".tensorboard", "mask_rcnn")

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def eval_mAP(test_path, json_path, checkpoint_dir, model_checkpoint=None):
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    inference_config = InferenceConfig()

    # Create the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            checkpoint_dir=checkpoint_dir,
                            tensorboard_dir=TENSORBOARD_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_checkpoint is None:
        model_path = model.find_last()
    else:
        checkpoints = next(os.walk(checkpoint_dir))[2]
        checkpoints = list(filter(lambda f: f.startswith(model_checkpoint), checkpoints))
        model_path = os.path.join(checkpoint_dir, checkpoints[0])

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    model.load_weights(model_path, by_name=True)

    dataset_val = CellsDataset()
    dataset_val.load_cells(test_ids)
    dataset_val.prepare()

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

        for j in range(r["masks"].shape[2]):
            # apply post-processing to mask
            r["masks"][:,:,j] = utils.mask_post_process(r["image"], r["masks"][:,:,j])

        r = utils.filter_result(r)

        # Compute AP @ different IoUs
        APs_img = []
        for thres in np.linspace(0.5, 0.95, 10):
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r["masks"], iou_threshold=thres)
            APs_img.append(AP)
        thresAP = np.mean(APs_img)
        eval_json.append({"img": dataset_val.image_info[i]["simple_path"], "AP": APs_img})
        APs.append(thresAP)

    with open(os.path.join(test_path, json_path), 'w') as fp:
        json.dump(eval_json, fp)
        
    return np.mean(APs)

if __name__ == "__main__":
    val_path='./data/stage1_val/'

    for i in range(1,32,10):
        checkpoint_path = "mask_rcnn_cells_{0:04}".format(i)
        json_path = "evals{0}.json".format(i)
        mAP = eval_mAP(val_path, json_path, checkpoint_dir=CHECKPOINT_DIR, model_checkpoint=checkpoint_path)
        print("mAP:", mAP)
    