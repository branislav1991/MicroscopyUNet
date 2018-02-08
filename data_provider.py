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

class TrainDataProviderResizeBinary(DataProvider):
    '''Provides binary segmentation with one class as mask and the other as background.
    This is definitely not optimal since a lot of times the algorithm cannot discriminate
    between 2 different cells overlapping.
    '''
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

class TrainDataProviderResizeMulticlass(DataProvider):
    '''Provides multiclass (semantic) segmentation by dividing the output into 3 classes.
    We use 0=inside region of the mask, 1=edge of the mask, 2=background.
    TODO: Optionally we want to support class weighting to balance dataset...
    '''

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
        self.Y = np.zeros((len(self.ids) * per_augmentation, img_height, img_width, 
            model.NUM_CLASSES), dtype=np.float32)

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
            mask_inner = np.zeros((img_height, img_width), dtype=np.float32)
            mask_edge = np.zeros_like(mask_inner)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_inner_ = io.imread(path + '/masks/' + mask_file)
                mask_inner_ = resize(mask_inner_, (img_height, img_width), mode='constant', 
                                            preserve_range=True)
                mask_inner = np.minimum(np.maximum(mask_inner, mask_inner_),1).astype(np.float32)
                mask_edge_ = cv2.morphologyEx(
                    mask_inner_, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                mask_edge = np.minimum(np.maximum(mask_edge, mask_edge_),1).astype(np.float32)

            mask_inner = np.logical_and(mask_inner, np.logical_not(mask_edge)).astype(np.float32)
            mask_background = np.logical_not(np.minimum(np.maximum(mask_edge, mask_inner),1)).astype(np.float32)
            self.Y[n*per_augmentation,:,:,0] = mask_inner
            self.Y[n*per_augmentation,:,:,1] = mask_edge
            self.Y[n*per_augmentation,:,:,2] = mask_background

            if augmentation is not None:
                imgs, masks = augment(img, self.Y[n*per_augmentation,...], augmentation)
                for i in range(0, count_augments(augmentation)):
                    self.X[n*per_augmentation + i+1,:,:,0] = imgs[i]
                    self.Y[n*per_augmentation + i+1,...] = masks[i]

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