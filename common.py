import os
import sys

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label

def create_folder(f):
    if not os.path.exists(f):
        os.makedirs(f)

def create_predicted_folders(ids):
    for pathar in ids:
        if not os.path.exists(pathar[0] + "/masks_predicted"):
            os.makedirs(pathar[0] + "/masks_predicted")

def load_train_images(train_path, img_height, img_width, img_channels):
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)
    sizes_train = []
    sys.stdout.flush()

    for n, pathar in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = pathar[0]
        id_ = pathar[1]
        img = io.imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
        sizes_train.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = io.imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                        preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    return X_train, Y_train, sizes_train, train_ids

def load_test_images(test_path, img_height, img_width, img_channels):
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]

    X_test = np.zeros((len(test_ids), img_height, img_width, img_channels), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, pathar in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = pathar[0]
        id_ = pathar[1]
        img = io.imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_test, sizes_test, test_ids

def IoU(labels, predictions):
    TP = (np.logical_and(np.logical_and(np.equal(labels, predictions), np.equal(labels,1)), np.equal(predictions, 1))).astype(float)
    FPandFN = (np.not_equal(labels, predictions)).astype(float)

    # compute mean
    TP = np.sum(TP, axis=(1,2,3))
    FPandFN = np.sum(FPandFN, axis=(1,2,3))
    return TP / (TP + FPandFN)

def mIoU(labels, predictions):
    return np.mean(IoU(labels, predictions))