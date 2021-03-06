import os
import sys
import math

from tqdm import tqdm
import numpy as np
from datetime import datetime
from skimage import io
from skimage.transform import resize, rescale, SimilarityTransform, warp, rotate
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import random

from models.unet.common import count_augments, preprocess, augment

class DataProvider():
    def __init__(self):
        self.i = 0
        self.ids = []
        self.sizes = []

    def get_sizes(self):
        return self.sizes

    def get_ids(self):
        return self.ids

    def reset(self):
        self.i = 0

class TrainDataProviderMulticlass(DataProvider):
    '''Provides multiclass (semantic) segmentation by dividing the output into 3 classes.
    We use 0=inside region of the mask, 1=edge of the mask, 2=background.
    TODO: Optionally we want to support class weighting to balance dataset...
    '''

    def __init__(self, num_img_channels, ids, shuffle=False, preprocessing=None, augmentation=None):
        DataProvider.__init__(self)

        self.ids = ids

        if augmentation is not None:
            per_augmentation = 1 + count_augments(augmentation)
        else:
            per_augmentation = 1

        # Get and resize train images and masks
        self.X = []
        self.Y = []
        #self.X = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, num_img_channels), dtype=np.float32)
        #self.Y = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)

        self.sizes = []
        sys.stdout.flush()

        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')[:,:,:3]
            self.sizes.append([img.shape[0], img.shape[1]])
            if preprocessing is not None:
                img = preprocess(img, preprocessing)

            self.X.append(np.reshape(img, (img.shape[0], img.shape[1], 1)))
            mask_inner = np.zeros_like(img, dtype=np.float32)
            #mask_edge = np.zeros_like(mask_inner)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_inner_ = io.imread(path + '/masks/' + mask_file)
                #mask_inner_ = resize(mask_inner_, (img_height, img_width), mode='constant', 
                #                            preserve_range=True)
                mask_inner = np.minimum(np.maximum(mask_inner, mask_inner_),1).astype(np.float32)
                #mask_edge_ = cv2.morphologyEx(
                #    mask_inner_, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
                #mask_edge = np.minimum(np.maximum(mask_edge, mask_edge_),1).astype(np.float32)

            #mask_edge = (mask_edge > 0).astype(np.float32)
            #mask_inner = np.logical_and(mask_inner, np.logical_not(mask_edge)).astype(np.float32)
            #mask_background = np.logical_not(np.minimum(np.maximum(mask_edge, mask_inner),1)).astype(np.float32)
            self.Y.append(mask_inner[...,None])
            #self.Y[n*per_augmentation,:,:,0] = mask_inner
            #self.Y[n*per_augmentation,:,:,1] = mask_edge
            #self.Y[n*per_augmentation,:,:,2] = mask_background

            if augmentation is not None:
                imgs, masks = augment(img, self.Y[n*per_augmentation,...], augmentation)
                for i in range(0, count_augments(augmentation)):
                    self.X.append(np.reshape(imgs[i], 
                        (imgs[i].shape[0], imgs[i].shape[1], 1)))
                    self.Y.append(masks[i])

        # shuffle if needed
        if shuffle == True:
            random.seed(3)
            self.X = random.shuffle(self.X)
            random.seed(3)
            self.Y = random.shuffle(self.Y)
            # perm = np.random.permutation(self.X.shape[0])
            # self.X = self.X[perm,...]
            # self.Y = self.Y[perm,...]

        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.num_elements():
            img = self.X[self.i][None,...]
            mask = self.Y[self.i][None,...]     
            self.i += 1
            return img, mask
        else:
            raise StopIteration()

    def num_elements(self):
        return len(self.X)

    def num_batches(self):
        return len(self.X)

    def get_true_Y(self):
        return self.Y

class TrainDataProviderTilingMulticlass(DataProvider):
    '''Provides multiclass (semantic) segmentation by dividing the output into 3 classes.
    We use 0=inside region of the mask, 1=edge of the mask, 2=background.
    This data provider randomly extracts tiles of model size from image. If the
    image is not large enough it is resized so that at least one tile can be extracted.
    '''

    def __init__(self, img_height, img_width, num_img_channels, ids, batch_size=2, shuffle=False, preprocessing=None, num_tiles=5):
        DataProvider.__init__(self)

        self.batch_size = batch_size
        self.ids = ids
        factor = 1.5

        # Get and tile train images and masks
        self.X = np.zeros((len(self.ids) * num_tiles, img_height, img_width, num_img_channels), dtype=np.float32)
        self.Y = np.zeros((len(self.ids) * num_tiles, img_height, img_width, 1), dtype=np.float32)

        self.sizes = []
        sys.stdout.flush()

        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')[:,:,:3]
            self.sizes.append([img.shape[0], img.shape[1]])
            if img.shape[0] < (factor*img_height) or img.shape[1] < (factor*img_width):
                scale_height = img.shape[0] / (2*img_height)
                scale_width = img.shape[1] / (2*img_width)
                scale = 1/max(scale_height, scale_width)
                img = rescale(img, scale)

            if preprocessing is not None:
                img = preprocess(img, preprocessing)

            mask_inner = np.zeros((self.sizes[-1][0], self.sizes[-1][1]), dtype=np.float32)
            #mask_edge = np.zeros_like(mask_inner)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_inner_ = io.imread(path + '/masks/' + mask_file)
                mask_inner = np.minimum(np.maximum(mask_inner, mask_inner_),1).astype(np.float32)
                #mask_edge_ = cv2.morphologyEx(
                #    mask_inner_, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
                #mask_edge = np.minimum(np.maximum(mask_edge, mask_edge_),1).astype(np.float32)

            #mask_edge = (mask_edge > 0).astype(np.float32)
            #mask_inner = np.logical_and(mask_inner, np.logical_not(mask_edge)).astype(np.float32)

            if mask_inner.shape[0] < (factor*img_height) or mask_inner.shape[1] < (factor*img_width):
                scale_height = mask_inner.shape[0] / (2*img_height)
                scale_width = mask_inner.shape[1] / (2*img_width)
                scale = 1/max(scale_height, scale_width)
                mask_inner = rescale(mask_inner, scale)

            # if mask_edge.shape[0] < (factor*img_height) or mask_edge.shape[1] < (factor*img_width):
            #     scale_height = img.shape[0] / (2*img_height)
            #     scale_width = img.shape[1] / (2*img_width)
            #     scale = 1/max(scale_height, scale_width)
            #     mask_edge = rescale(mask_edge, scale)
            
            ms = datetime.now().microsecond
            rnd = np.random.RandomState(seed=ms)
            tiles_img = extract_patches_2d(img, (img_height, img_width), num_tiles, random_state=rnd)
            rnd = np.random.RandomState(seed=ms)
            tiles_mask_inner = extract_patches_2d(mask_inner, (img_height, img_width), num_tiles, random_state=rnd)
            #rnd = np.random.RandomState(seed=ms)
            #tiles_mask_edge = extract_patches_2d(mask_edge, (img_height, img_width), num_tiles, random_state=rnd)

            self.X[n*num_tiles:(n+1)*num_tiles,...] = np.reshape(tiles_img, (num_tiles, img_height, img_width, num_img_channels))
            
            self.Y[n*num_tiles:(n+1)*num_tiles,:,:,0] = tiles_mask_inner
            #self.Y[n*num_tiles:(n+1)*num_tiles,:,:,1] = tiles_mask_edge

        self.Y[self.Y > 0] = 1.0 # make sure our mask is binary

        # shuffle if needed
        if shuffle == True:
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm,...]
            self.Y = self.Y[perm,...]

        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.num_batches():
            begin = self.i * self.batch_size
            end = (self.i + 1) * self.batch_size
            img = self.X[begin:end,...]
            mask = self.Y[begin:end,...]
                
            self.i += 1
            return img, mask
        else:
            raise StopIteration()

    def num_elements(self):
        return self.X.shape[0]

    def num_batches(self):
        return math.ceil(float(self.X.shape[0]) / self.batch_size)

    def get_true_Y(self):
        return self.Y

class TestDataProvider(DataProvider):
    ''' This data provider optionally provides test data without resizing. However,
    this is currently not supported for the network architecture of UNet. It also cannot be
    reset as the training data provider can.'''
    def __init__(self, ids, preprocessing=None):
        DataProvider.__init__(self)

        self.ids = ids

        self.X = []
        self.sizes = []
        sys.stdout.flush()
        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')[:,:,:3]
            self.sizes.append([img.shape[0], img.shape[1]])

            if preprocessing is not None:
                img = preprocess(img, preprocessing)

            self.X.append(np.reshape(img, (*img.shape, 1)))

        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.X):
            img = self.X[self.i]
            self.i += 1
            return np.reshape(img, [1, *img.shape])
        else:
            raise StopIteration()

    def num_elements(self):
        return len(self.X)
        
    def num_batches(self):
        return len(self.X)