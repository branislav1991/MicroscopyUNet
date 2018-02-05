import os
import sys
import math

import numpy as np
import tensorflow as tf

from skimage import io
from skimage.transform import resize, SimilarityTransform, warp, rotate
from skimage.color import rgb2lab, rgb2gray
from elastic_transform import elastic_transform
import cv2
from scipy.ndimage.interpolation import map_coordinates

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
    if augmentation.get('rotate_rnd') is not None:
        for i in range(0, augmentation['rotate_rnd']):
            rnd = np.random.rand(1) * 90 - 45
            imgs.append(rotate(img, rnd))
            masks.append(rotate(mask, rnd, order=0))

    if augmentation.get('elastic_rnd') is not None:
        for i in range(0, augmentation['elastic_rnd']):
            M, ind = elastic_transform(img, 30, 30, 30)
            img = cv2.warpAffine(img, M, img.shape, borderMode=cv2.BORDER_REFLECT_101)
            img = map_coordinates(img, ind, order=2, mode='reflect').reshape(img.shape)
            imgs.append(img)

            mask = cv2.warpAffine(mask, M, img.shape, borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_NEAREST)
            mask = map_coordinates(mask, ind, order=0, mode='reflect').reshape(img.shape)
            masks.append(mask)

    if augmentation.get('resize_rnd') is not None:
        for i in range(0, augmentation['resize_rnd']):
            # random zoom from 0.7 to 1.3
            rnd = np.random.rand(1) * 0.6 + 0.7
            tform = SimilarityTransform(scale=rnd)    
            imgs.append(warp(img, tform))
            masks.append(warp(mask, tform, order=0))

    return imgs, masks

def count_augments(augmentation):
    n_augments = 0
    for a in augmentation:
        n_augments = n_augments + augmentation[a]
    return n_augments

def IoU(labels, predictions):
    TP = (np.logical_and(np.logical_and(np.equal(labels, predictions), np.equal(labels,1)), np.equal(predictions, 1))).astype(float)
    FPandFN = (np.not_equal(labels, predictions)).astype(float)

    # compute mean
    TP = np.sum(TP, axis=(1,2,3))
    FPandFN = np.sum(FPandFN, axis=(1,2,3))
    return TP / (TP + FPandFN)

def mIoU(labels, predictions):
    return np.mean(IoU(labels, predictions))