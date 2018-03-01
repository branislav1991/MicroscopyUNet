import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans

TRAIN_PATH=".\\data\\stage1_train\\"
IMG_WIDTH = 256
IMG_HEIGHT = 256

def train_kmeans():
    train_ids = next(os.walk(TRAIN_PATH))
    train_ids = [train_ids[0] + d + '\\images\\' + d + ".png" for d in train_ids[1]]
    
    imgs = np.zeros((len(train_ids), IMG_HEIGHT*IMG_WIDTH*3))
    for i, id_ in enumerate(train_ids):
        img_ = imread(id_)[:,:,:3]
        img_ = resize(img_, (IMG_HEIGHT, IMG_WIDTH))
        imgs[i,:] = img_.flatten()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(imgs)
    labels = kmeans.labels_

    return labels

if __name__ == "__main__":
    labels = train_kmeans()
    pass