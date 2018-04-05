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

def ensemble_mask_rcnn(test_ids, test_path, checkpoint_dir, angles):
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
        angles_masks = []
        angles_scores = [] 
        img = dataset_test.load_image(id)
        for angle in angles:
            if abs(angle) > 0:
                center=tuple(np.array(img.shape[1::-1])//2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
                inv_rot_mat = cv2.invertAffineTransform(rot_mat)

                img_rot = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
                detection = model.detect([img_rot], verbose=1)
                masks = detection[0]["masks"]
                for i in range(masks.shape[2]):
                    masks[:,:,i] = cv2.warpAffine(masks[:,:,i], inv_rot_mat, masks.shape[1::-1], flags=cv2.INTER_NEAREST)
            else:
                detection = model.detect([img], verbose=1)

            angles_masks.append(detection[0]["masks"])
            if abs(angle) > 0:
                detection[0]["scores"] = detection[0]["scores"] - 0.5
            angles_scores.append(detection[0]["scores"])

        angles_masks = np.concatenate(angles_masks, axis=2)
        angles_scores = np.concatenate(angles_scores)
        idx_retained = utils.non_max_suppression_masks(angles_masks, angles_scores, inference_config.ENSEMBLE_MASK_NMS_THRESHOLD)
        angles_masks = angles_masks[:,:,idx_retained]

        path = os.path.join(dataset_test.image_info[id]["simple_path"], "masks_predicted")
        for j in range(angles_masks.shape[2]):
            # apply post-processing to mask
            angles_masks[:,:,j] = utils.mask_post_process(img, angles_masks[:,:,j])
            io.imsave("{0}/mask_{1}.tif".format(path, j), angles_masks[:,:,j] * 255)

        #result = utils.filter_result(result)

    print("Done!")

if __name__ == "__main__":
    angles = [0, 30]

    train_path='./data/stage1_val/'
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
    ensemble_mask_rcnn(train_ids, train_path, CHECKPOINT_DIR, angles)

    test_path='./data/stage1_test/'
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
    ensemble_mask_rcnn(test_ids, test_path, CHECKPOINT_DIR, angles)
    
