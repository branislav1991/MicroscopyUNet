import os
import sys
import math

from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.transform import resize, SimilarityTransform, warp, rotate

from common import count_augments, preprocess, augment

# TODO: enable batch providing etc. etc. from this class

class DataProvider_old():
    def __init__(self, model, train_path=None, test_path=None):
        if train_path is None:
            #self.train_path = './data/stage1_train/'
            self.train_path = './data/stage1_train_small/'
        else:
            self.train_path = train_path

        if test_path is None:
            self.test_path = './data/stage1_test/'
        else:
            self.test_path = test_path

        self.model = model
    
    def load_train_images_resize(self, preprocessing=None, augmentation=None):
        # note: sizes_train is only correct if we do not do any augmentation (irrelevant for testing)
        train_ids = next(os.walk(self.train_path))
        train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

        if augmentation is not None:
            per_augmentation = 1 + count_augments(augmentation)
        else:
            per_augmentation = 1

        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

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
                for i in range(0, count_augments(augmentation)):
                    X_train[n*per_augmentation + i+1,:,:,0] = imgs[i]
                    Y_train[n*per_augmentation + i+1,:,:,0] = masks[i]

        return X_train, Y_train, sizes_train, train_ids

    # def load_train_images_rnd_tile(self, min_size=512, preprocessing=None, augmentation=None):
    #     # note: sizes_train is only correct if we do not do any augmentation (irrelevant for testing)
    #     train_ids = next(os.walk(self.train_path))
    #     train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

    #     if augmentation is not None:
    #         per_augmentation = 1 + count_augments(augmentation)
    #     else:
    #         per_augmentation = 1

    #     img_height = self.model.IMG_HEIGHT
    #     img_width = self.model.IMG_WIDTH

    #     # Get and resize train images and masks
    #     X_train = np.zeros((len(train_ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
    #     Y_train = np.zeros((len(train_ids) * per_augmentation, img_height, img_width, 1), dtype=np.float32)
    #     sizes_train = []
    #     sys.stdout.flush()

    #     for n, pathar in tqdm(enumerate(train_ids), total=len(train_ids)):
    #         path = pathar[0]
    #         id_ = pathar[1]
    #         img = io.imread(path + '/images/' + id_ + '.png')
    #         sizes_train.append([img.shape[0], img.shape[1]])
    #         img = resize(img, (img_height, img_width), mode='constant')[:,:,:3]
    #         if preprocessing is not None:
    #             img = preprocess(img, preprocessing)

    #         X_train[n*per_augmentation] = np.reshape(img, (img_height, img_width, 1))
    #         mask = np.zeros((img_height, img_width, 1), dtype=np.float32)
    #         for mask_file in next(os.walk(path + '/masks/'))[2]:
    #             mask_ = io.imread(path + '/masks/' + mask_file)
    #             mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
    #                                         preserve_range=True), axis=-1)
    #             mask = np.minimum(np.maximum(mask, mask_),1).astype(np.float32)
    #         Y_train[n*per_augmentation] = mask

    #         if augmentation is not None:
    #             imgs, masks = augment(img, mask[:,:,0], augmentation)
    #             for i in range(0, count_augments(augmentation)):
    #                 X_train[n*per_augmentation + i+1,:,:,0] = imgs[i]
    #                 Y_train[n*per_augmentation + i+1,:,:,0] = masks[i]

    #     return X_train, Y_train, sizes_train, train_ids

    def load_test_images(self, res=True, preprocessing=None):
        test_ids = next(os.walk(self.test_path))
        test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]

        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

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

    def train_val_split(self, X, Y, p):
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

    def shuffle_dataset(self, X_train, Y_train):
        data_size = X_train.shape[0]
        p = np.random.permutation(data_size)
        return X_train[p,...], Y_train[p,...]

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

class TrainDataProviderResize(DataProvider):
    pass

class TrainDataProviderTile(DataProvider):
    pass

class TestDataProvider(DataProvider):
    def __init__(self, model, test_path='./data/stage1_test/', res=True, preprocessing=None):
        DataProvider.__init__(self, model)
        self.model = model

        self.ids = next(os.walk(test_path))
        self.ids = [[self.ids[0] + d,d] for d in self.ids[1]]

        img_height = self.model.IMG_HEIGHT
        img_width = self.model.IMG_WIDTH

        #self.X = np.zeros((len(self.self.test_ids), img_height, img_width, 1), dtype=np.float32)
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
    