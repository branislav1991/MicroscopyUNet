import os
import sys
import math

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage import io
from skimage.transform import resize, SimilarityTransform, warp, rotate
from skimage.color import rgb2lab, rgb2gray
from skimage.util import apply_parallel

import matplotlib.pyplot as plt

def create_folder(f):
    if not os.path.exists(f):
        os.makedirs(f)

def create_predicted_folders(ids):
    for pathar in ids:
        if not os.path.exists(pathar[0] + "/masks_predicted"):
            os.makedirs(pathar[0] + "/masks_predicted")

def preprocess(img, preprocessing):
    if img.shape[2] == 3: # some preprocessing is only valid for color images
        if 'Lab' in preprocessing:
            # just extract L channel from Lab / invert to normalize stained images
            img = rgb2lab(img)
            img = img[:,:,0]
            img = img / np.amax(img)
            img_mean = img.mean()
            if img_mean > 0.5:
                img = np.subtract(1.0, img)

        else:
            img = rgb2gray(img)

    return img

def augment(img, mask, augmentation):
    # returns a list of augmented images
    # DOES NOT return original image!
    imgs = []
    masks = []
    if 'rotate_rnd' in augmentation:
        # random rotation from -45 to 45 degrees
        #rnd = np.random.rand(1) * (math.pi/2) - (math.pi/4)
        #tform = SimilarityTransform(scale=1, rotation=rnd)    
        #imgs.append(warp(img, tform))
        #masks.append(warp(mask, tform, order=0))
        rnd = np.random.rand(1) * 90 - 45
        imgs.append(rotate(img, rnd))
        masks.append(rotate(mask, rnd, order=0))

    if 'resize_rnd' in augmentation:
        # random zoom from 0.7 to 1.3
        rnd = np.random.rand(1) * 0.6 + 0.7
        tform = SimilarityTransform(scale=rnd)    
        imgs.append(warp(img, tform))
        masks.append(warp(mask, tform, order=0))

    return imgs, masks


def load_train_images(train_path, img_height, img_width, preprocessing=None, augmentation=None):
    # note: sizes_train is only correct if we do not do any augmentation (irrelevant for testing)
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

    per_augmentation = 1 + len(augmentation)

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
    Y_train = np.zeros((len(train_ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
    sizes_train = []
    sys.stdout.flush()

    for n, pathar in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = pathar[0]
        id_ = pathar[1]
        img = io.imread(path + '/images/' + id_ + '.png')
        sizes_train.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant')[:,:,:3]
        if preprocessing is not None:
            img = preprocess(img, preprocessing)

        X_train[n*per_augmentation] = np.reshape(img, (img_height, img_width, 1))
        mask = np.zeros((img_height, img_width, 1), dtype=np.float32)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = io.imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                        preserve_range=True), axis=-1)
            mask = np.minimum(np.maximum(mask, mask_),1).astype(np.float32)
        Y_train[n*per_augmentation] = mask

        if augmentation is not None:
            imgs, masks = augment(img, mask[:,:,0], augmentation)
            for i in range(0, len(augmentation)):
                X_train[n*per_augmentation + i+1,:,:,0] = imgs[i]
                Y_train[n*per_augmentation + i+1,:,:,0] = masks[i]

    return X_train, Y_train, sizes_train, train_ids

def load_test_images(test_path, img_height, img_width, preprocessing=None):
    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]

    X_test = np.zeros((len(test_ids), img_height, img_width, 1), dtype=np.float32)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, pathar in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = pathar[0]
        id_ = pathar[1]
        img = io.imread(path + '/images/' + id_ + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant')[:,:,:3]

        if preprocessing is not None:
            img = preprocess(img, preprocessing)

        X_test[n] = np.reshape(img, (img_height, img_width, 1))

    return X_test, sizes_test, test_ids

def train_val_split(X, Y, p):
    if p > 1.0:
        ValueError("p has to be smaller than 1 but is {0}".format(p))

    data_size = X.shape[0]
    val_part_size = math.floor(data_size * p)
    train_part_size = data_size - val_part_size

    X_train = X[:train_part_size, ...]
    Y_train = Y[:train_part_size, ...]

    X_val = X[-val_part_size:, ...]
    Y_val = Y[-val_part_size:, ...]

    return X_train, Y_train, X_val, Y_val

def IoU(labels, predictions):
    TP = (np.logical_and(np.logical_and(np.equal(labels, predictions), np.equal(labels,1)), np.equal(predictions, 1))).astype(float)
    FPandFN = (np.not_equal(labels, predictions)).astype(float)

    # compute mean
    TP = np.sum(TP, axis=(1,2,3))
    FPandFN = np.sum(FPandFN, axis=(1,2,3))
    return TP / (TP + FPandFN)

def mIoU(labels, predictions):
    return np.mean(IoU(labels, predictions))