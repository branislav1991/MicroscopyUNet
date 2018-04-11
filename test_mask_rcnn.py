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

BBOX_CLASS_FNAME = "roi_class.json"

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.7

def test_mask_rcnn(test_ids, test_path, checkpoint_dir):
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

    roi_class = []

    print('Loading training images ... ')
    create_predicted_folders(test_ids)

    dataset_test = CellsDataset()
    dataset_test.load_cells(test_ids)
    dataset_test.prepare()

    # Evaluate dataset to obain average mask sizes
    print('Evaluating dataset ... ')
    results = []
    for id in dataset_test.image_ids:
        img = dataset_test.load_image(id)
        results.append(model.detect([img], verbose=0, scale=1))

    # Measure average mask sizes and append to list
    avg_mask_sizes = []
    for i, res in enumerate(results):
        #avg_mask_sizes.append(max(np.mean(np.sum(res[0]["masks"], axis=(0,1))) / 1500.0, 1.0))
        avg_mask_sizes.append(1.0)
        print("The average size of masks is {0}".format(avg_mask_sizes[-1]))

    results = []
    for i, id in enumerate(dataset_test.image_ids):
        img = dataset_test.load_image(id)
        results.append(model.detect([img], verbose=0, scale=avg_mask_sizes[i]))

    print("Saving generated masks ...")
    for i, res in tqdm(enumerate(results), total=len(results)):
        path = os.path.join(dataset_test.image_info[i]["simple_path"], "masks_predicted")
        for j in range(res[0]["masks"].shape[2]):
            # apply post-processing to mask
            res[0]["masks"][:,:,j] = utils.mask_post_process(res[0]["image"], res[0]["masks"][:,:,j])

        res[0] = utils.filter_result(res[0])

        average_mask_size = np.mean(np.sum(res[0]["masks"], axis=(0,1)))

        # save mask
        for j in range(res[0]["masks"].shape[2]):
            io.imsave("{0}/mask_{1}.tif".format(path, j), res[0]["masks"][:,:,j] * 255)

        # also save other textual information retrieved by the CNN
        class_ids = res[0]["class_ids"].tolist()
        class_names = [x["name"] for x in dataset_test.class_info]
        roi_class.append({"img": dataset_test.image_info[i]["simple_path"], "rois": [[i, tuple(r)] for (i,r) in enumerate(res[0]["rois"].tolist())], 
                                "class_ids": class_ids, "class_names": class_names,
                                "scores": res[0]["scores"].tolist()})

    with open(os.path.join(test_path, BBOX_CLASS_FNAME), 'w') as fp:
        json.dump(roi_class, fp)
    print("Done!")

if __name__ == "__main__":
    train_path='./data/stage1_val/'
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
    test_mask_rcnn(train_ids, train_path, CHECKPOINT_DIR)

    test_path='./data/stage1_test/'
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
    test_mask_rcnn(test_ids, test_path, CHECKPOINT_DIR)
    