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

inference_config = InferenceConfig()


# Create the model in inference mode
print("Initializing model in inference mode ... ")
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          checkpoint_dir=CHECKPOINT_DIR,
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



# Training dataset
roi_class_train = []

print('Loading training images and masks ... ')
train_path='.\\data\\stage1_train_small\\'
train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
create_predicted_folders(train_ids)

dataset_train = CellsDataset()
dataset_train.load_cells(train_ids)
dataset_train.prepare()

# Evaluate training dataset
print('Evaluating training dataset ... ')
results = []
for id in dataset_train.image_ids:
    img = dataset_train.load_image(id)
    results.append(model.detect([img], verbose=1))

print("Saving generated masks ...")
for i, res in tqdm(enumerate(results), total=len(results)):
    path = os.path.join(dataset_train.image_info[i]["simple_path"], "masks_predicted")
    mask = res[0]["masks"]
    for j in range(mask.shape[2]):
        io.imsave("{0}/mask_{1}.tif".format(path, j), mask[:,:,j] * 255)

    # also save other textual information retrieved by the CNN
    class_ids = res[0]["class_ids"].tolist()
    class_names = [x["name"] for x in dataset_train.class_info]
    roi_class_train.append({"img": dataset_train.image_info[i]["simple_path"], "rois": [[i, tuple(r)] for (i,r) in enumerate(res[0]["rois"].tolist())], 
                            "class_ids": class_ids, "class_names": class_names,
                            "scores": res[0]["scores"].tolist()})

with open(os.path.join(train_path, BBOX_CLASS_FNAME), 'w') as fp:
    json.dump(roi_class_train, fp)
print("Done!")

# Testing dataset
roi_class_test = []

test_path='.\\data\\stage1_test\\'
test_ids = next(os.walk(test_path))
test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
create_predicted_folders(test_ids)

dataset_test = CellsDataset()
dataset_test.load_cells(test_ids)
dataset_test.prepare()

# Evaluate testing dataset
print('Evaluating testing dataset ... ')
results = []
for id in dataset_test.image_ids:
    img = dataset_test.load_image(id)
    results.append(model.detect([img], verbose=1))

print("Saving generated masks ...")
for i, res in tqdm(enumerate(results), total=len(results)):
    path = os.path.join(dataset_test.image_info[i]["simple_path"], "masks_predicted")
    mask = res[0]["masks"]
    for j in range(mask.shape[2]):
        io.imsave("{0}/mask_{1}.tif".format(path, j), mask[:,:,j] * 255)

    # also save other textual information retrieved by the CNN
    class_ids = res[0]["class_ids"].tolist()
    class_names = [x["name"] for x in dataset_test.class_info]
    roi_class_test.append({"img": dataset_test.image_info[i]["simple_path"], "rois": [[i, tuple(r)] for (i,r) in enumerate(res[0]["rois"].tolist())], 
                            "class_ids": class_ids, "class_names": class_names,
                            "scores": res[0]["scores"].tolist()})

with open(os.path.join(test_path, BBOX_CLASS_FNAME), 'w') as fp:
    json.dump(roi_class_test, fp)
print("Done!")



# # Test on a random image
# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, inference_config, 
#                            image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
#                             dataset_train.class_names, figsize=(8, 8))


# results = model.detect([original_image], verbose=1)

# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                             dataset_val.class_names, r['scores'], ax=get_ax())



# # Compute VOC-Style mAP @ IoU=0.5
# # Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_val.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#         modellib.load_image_gt(dataset_val, inference_config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps =\
#         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)
    
# print("mAP: ", np.mean(APs))