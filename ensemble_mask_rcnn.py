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

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def ensemble_mask_rcnn(test_ids, test_path, checkpoint_dir, augments):
    inference_config = InferenceConfig()

    # Create the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            checkpoint_dir=checkpoint_dir,
                            tensorboard_dir=TENSORBOARD_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    roi_class = []

    create_predicted_folders(test_ids)

    dataset_test = CellsDataset()
    dataset_test.load_cells(test_ids)
    dataset_test.prepare()

    # Process all images
    for id in dataset_test.image_ids:
        img = dataset_test.load_image(id)
        # baseline detection
        detection = model.detect([img], verbose=1)
        baseline_masks = detection[0]["masks"]
        baseline_scores = detection[0]["scores"]

        # augments detection
        augment_masks = []
        augment_scores = [] 
        for augment in augments:
            if augment["name"] == "angle":
                width = img.shape[1]
                height = img.shape[0]
                width_big = width * 2
                height_big = height * 2

                img_big = cv2.resize(img, (width_big, height_big))
                center=tuple(np.array(img_big.shape[1::-1])//2)
                rot_mat = cv2.getRotationMatrix2D(center, augment["angle"], 0.5)
                inv_rot_mat = cv2.invertAffineTransform(rot_mat)

                img_rot = cv2.warpAffine(img_big, rot_mat, img_big.shape[1::-1], flags=cv2.INTER_LINEAR)
                detection = model.detect([img_rot], verbose=1)
                masks = detection[0]["masks"]
                masks_small = []
                for i in range(masks.shape[2]):
                    masks[:,:,i] = cv2.warpAffine(masks[:,:,i], inv_rot_mat, masks.shape[1::-1], flags=cv2.INTER_NEAREST)
                    masks_small.append(cv2.resize(masks[:,:,i], (width, height)))
                detection[0]["masks"] = np.stack(masks_small, axis=2)
            elif augment["name"] == "fliplr":
                img_flipped = np.fliplr(img) 
                detection = model.detect([img_flipped], verbose=1)
                masks = detection[0]["masks"]
                for i in range(masks.shape[2]):
                    masks[:,:,i] = np.fliplr(masks[:,:,i]) 

            augment_masks.append(detection[0]["masks"])
            augment_scores.append(detection[0]["scores"])

        if len(augment_masks) > 0:
            augment_masks = np.concatenate(augment_masks, axis=2)
            augment_scores = np.concatenate(augment_scores)
            #idx_retained = utils.non_max_suppression_masks(augment_masks, augment_scores, inference_config.ENSEMBLE_MASK_NMS_THRESHOLD)
            #augment_masks = augment_masks[:,:,idx_retained]
            #augment_scores = augment_scores[idx_retained]

            #idx_retained = utils.suppress_augments(baseline_masks, augment_masks, inference_config.AUGMENT_REMOVAL_THRESHOLD)
            #augment_masks = augment_masks[:,:,idx_retained]
            #final_masks = np.concatenate([baseline_masks, augment_masks], axis=2)
            final_masks = augment_masks
        else:
            final_masks = baseline_masks

        path = os.path.join(dataset_test.image_info[id]["simple_path"], "masks_predicted")
        for j in range(final_masks.shape[2]):
            # apply post-processing to mask
            final_masks[:,:,j] = utils.mask_post_process(img, final_masks[:,:,j])
            io.imsave("{0}/mask_{1}.tif".format(path, j), final_masks[:,:,j] * 255)

        #result = utils.filter_result(result)

    print("Done!")

if __name__ == "__main__":
    #augments = [{"name": "angle", "angle": 45}]
    augments = [{"name": "fliplr"}]

    train_path='./data/stage1_val/'
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
    ensemble_mask_rcnn(train_ids, train_path, CHECKPOINT_DIR, augments)

    test_path='./data/stage1_test/'
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
    ensemble_mask_rcnn(test_ids, test_path, CHECKPOINT_DIR, augments)

