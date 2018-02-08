import os
import sys
import math

from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.transform import resize, SimilarityTransform, warp, rotate
import cv2

from common import count_augments, preprocess, augment

class DataProvider():
    def __init__(self, model):
        self.model = model
        self.i = 0
        self.ids = []
        self.sizes = []

    def get_sizes(self):
        return self.sizes

    def get_ids(self):
        return self.ids

    def reset(self):
        self.i = 0

class TrainDataProviderTile(DataProvider):
    pass

class TrainDataProviderResize(DataProvider):
    def __init__(self, model, ids, batch_size=2, shuffle=False, preprocessing=None, augmentation=None):
        DataProvider.__init__(self, model)

        self.batch_size = batch_size
        self.ids = ids

        # note: sizes_train is only correct if we do not do any augmentation (irrelevant for testing)
        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

        if augmentation is not None:
            per_augmentation = 1 + count_augments(augmentation)
        else:
            per_augmentation = 1

        # Get and resize train images and masks
        self.X = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
        self.Y = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
        self.sizes = []
        sys.stdout.flush()

        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')
            self.sizes.append([img.shape[0], img.shape[1]])
            img = resize(img, (img_height, img_width), mode='constant')[:,:,:3]
            if preprocessing is not None:
                img = preprocess(img, preprocessing)

            self.X[n*per_augmentation] = np.reshape(img, (img_height, img_width, 1))
            mask = np.zeros((img_height, img_width, 1), dtype=np.float32)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = io.imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                            preserve_range=True), axis=-1)
                mask = np.minimum(np.maximum(mask, mask_),1).astype(np.float32)
            self.Y[n*per_augmentation] = mask

            if augmentation is not None:
                imgs, masks = augment(img, [mask[:,:,0]], augmentation)
                for i in range(0, count_augments(augmentation)):
                    self.X[n*per_augmentation + i+1,:,:,0] = imgs[i]
                    self.Y[n*per_augmentation + i+1,:,:,0] = masks[0][i]

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

class TrainDataProviderResizeWithEdge(DataProvider):
    def __init__(self, model, ids, batch_size=2, shuffle=False, preprocessing=None, augmentation=None):
        DataProvider.__init__(self, model)

        self.batch_size = batch_size

        self.ids = ids

        # note: sizes_train is only correct if we do not do any augmentation (irrelevant for testing)
        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

        if augmentation is not None:
            per_augmentation = 1 + count_augments(augmentation)
        else:
            per_augmentation = 1

        # Get and resize train images and masks
        self.X = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
        self.Y = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
        self.Y_edge = np.zeros(self.Y.shape, dtype=np.float32)

        self.sizes = []
        sys.stdout.flush()

        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')
            self.sizes.append([img.shape[0], img.shape[1]])
            img = resize(img, (img_height, img_width), mode='constant')[:,:,:3]
            if preprocessing is not None:
                img = preprocess(img, preprocessing)

            self.X[n*per_augmentation] = np.reshape(img, (img_height, img_width, 1))
            mask = np.zeros((img_height, img_width, 1), dtype=np.float32)
            mask_edge = np.zeros(mask.shape, dtype=np.float32)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = io.imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                            preserve_range=True), axis=-1)
                mask = np.minimum(np.maximum(mask, mask_),1).astype(np.float32)
                mask_edge_ = cv2.morphologyEx(
                    mask_, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                mask_edge = np.minimum(np.maximum(mask_edge, mask_edge_[:,:,None]),1).astype(np.float32)
            self.Y[n*per_augmentation] = mask
            self.Y_edge[n*per_augmentation] = mask_edge

            if augmentation is not None:
                imgs, masks = augment(img, [mask[:,:,0], mask_edge[:,:,0]], augmentation)
                for i in range(0, count_augments(augmentation)):
                    self.X[n*per_augmentation + i+1,:,:,0] = imgs[i]
                    self.Y[n*per_augmentation + i+1,:,:,0] = masks[0][i]
                    self.Y_edge[n*per_augmentation + i+1,:,:,0] = masks[1][i]

        # shuffle if needed
        perm = np.random.permutation(self.X.shape[0])
        self.X = self.X[perm,...]
        self.Y = self.Y[perm,...]
        self.Y_edge = self.Y_edge[perm,...]

        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.num_batches():
            begin = self.i * self.batch_size
            end = (self.i + 1) * self.batch_size
            img = self.X[begin:end,...]
            #mask = self.Y[begin:end,...]
            mask = self.Y_edge[begin:end,...]
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

    def get_true_Y_edge(self):
        return self.Y_edge

class TestDataProvider(DataProvider):
    ''' This data provider optionally provides test data without resizing. However,
    this is currently not supported for the network architecture of UNet. It also cannot be
    reset as the training data provider can.'''
    def __init__(self, model, ids, res=True, preprocessing=None):
        DataProvider.__init__(self, model)

        self.ids = ids

        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

        self.X = []
        self.sizes = []
        sys.stdout.flush()
        for n, pathar in tqdm(enumerate(self.ids), total=len(self.ids)):
            path = pathar[0]
            id_ = pathar[1]
            img = io.imread(path + '/images/' + id_ + '.png')
            self.sizes.append([img.shape[0], img.shape[1]])

            img = img[:,:,:3]

            if res == True:
                img = resize(img, (img_height, img_width), mode='constant')

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