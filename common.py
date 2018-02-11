import os
import sys
import math

import numpy as np
import tensorflow as tf

from skimage import io
from skimage.transform import resize, SimilarityTransform, warp, rotate
from skimage.color import rgb2lab, rgb2gray
from skimage.filters import threshold_local
from elastic_transform import elastic_transform
import cv2
from scipy.ndimage.interpolation import map_coordinates

import matplotlib.pyplot as plt

SEGMENTATION_THRESHOLD = 0.5

def create_folder(f):
    if not os.path.exists(f):
        os.makedirs(f)

def create_predicted_folders(ids):
    for pathar in ids:
        if not os.path.exists(pathar[0] + "/masks_predicted"):
            os.makedirs(pathar[0] + "/masks_predicted")

def preprocess(img, preprocessing):
    if img.shape[2] > 1: # some preprocessing is only valid for color images
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

def augment(img, masks, augmentation):
    '''Augments the data that is available already. Can process one image and
    an arbitrary number of masks at the same time.
    Returns a list of augmented images and masks.
    Does not return original image.'''
    imgs_augmented = []
    masks_augmented = []
    
    if augmentation.get('rotate_rnd') is not None:
        for i in range(0, augmentation['rotate_rnd']):
            rnd = np.random.rand(1) * 90 - 45
            imgs_augmented.append(rotate(img, rnd))
            masks_augmented.append(np.zeros_like(masks))
            for j in range(0,masks.shape[2]):
                masks_augmented[-1][:,:,j] = rotate(m, rnd, order=0)

    if augmentation.get('elastic_rnd') is not None:
        for i in range(0, augmentation['elastic_rnd']):
            M, ind = elastic_transform(img, 30, 30, 30)
            img = cv2.warpAffine(img, M, img.shape, borderMode=cv2.BORDER_REFLECT_101)
            img = map_coordinates(img, ind, order=2, mode='reflect').reshape(img.shape)
            imgs_augmented.append(img)
            masks_augmented.append(np.zeros_like(masks))
            for j in range(0,masks.shape[2]):
                m = cv2.warpAffine(masks[:,:,j], M, img.shape, borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_NEAREST)
                m = map_coordinates(m, ind, order=0, mode='reflect').reshape(img.shape)
                masks_augmented[-1][:,:,j] = m

    if augmentation.get('resize_rnd') is not None:
        for i in range(0, augmentation['resize_rnd']):
            # random zoom from 0.7 to 1.3
            rnd = np.random.rand(1) * 0.6 + 0.7
            tform = SimilarityTransform(scale=rnd)    
            imgs_augmented.append(warp(img, tform))
            masks_augmented.append(np.zeros_like(masks))
            for j in range(0,masks.shape[2]):
                masks_augmented[-1][:,:,j] = warp(m, tform, order=0)

    return imgs_augmented, masks_augmented

def count_augments(augmentation):
    n_augments = 0
    for a in augmentation:
        n_augments = n_augments + augmentation[a]
    return n_augments

def IoU(labels, predictions):
    TP = (np.logical_and(np.logical_and(np.equal(labels, predictions), np.equal(labels,True)), np.equal(predictions, True))).astype(float)
    FPandFN = (np.not_equal(labels, predictions)).astype(float)

    # compute mean
    TP = np.sum(TP, axis=(1,2))
    FPandFN = np.sum(FPandFN, axis=(1,2))
    return TP / (TP + FPandFN)

def mIoU(labels, predictions):
    return np.mean(IoU(labels, predictions))

def adaptive_threshold(imgs):
    imgs_thrsh = np.zeros(imgs.shape, dtype=bool)
    for i in range(0, imgs.shape[0]):
        imgs_thrsh[i,:,:,0] = np.greater(imgs[i,:,:,0], threshold_local(imgs[i,:,:,0], 13))

    return imgs_thrsh
        