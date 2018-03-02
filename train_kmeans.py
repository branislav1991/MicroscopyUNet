import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans

TRAIN_PATH=".\\data\\stage1_train\\"
IMG_WIDTH = 256
IMG_HEIGHT = 256

def train_kmeans(train_path):
    train_ids = next(os.walk(train_path))
    train_ids = [train_ids[0] + d + '\\images\\' + d + ".png" for d in train_ids[1]]
    
    imgs = np.zeros((len(train_ids), IMG_HEIGHT*IMG_WIDTH*3))
    for i, id_ in enumerate(train_ids):
        img_ = imread(id_)[:,:,:3]
        img_ = resize(img_, (IMG_HEIGHT, IMG_WIDTH))
        imgs[i,:] = img_.flatten()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(imgs)
    labels = kmeans.labels_

    clustered = zip(train_ids, labels)
    clustered = sorted(clustered, key=lambda x: x[1])

    clusters = []
    clusters.append([x[0] for x in clustered if x[1]==0])
    clusters.append([x[0] for x in clustered if x[1]==1])
    clusters.append([x[0] for x in clustered if x[1]==2])

    plt.subplot(2,2,1)
    plt.imshow(imread(clusters[0][1])[:,:,:3])

    plt.subplot(2,2,2)
    plt.imshow(imread(clusters[1][1])[:,:,:3])

    plt.subplot(2,2,3)
    plt.imshow(imread(clusters[2][1])[:,:,:3])

    plt.show()
    return clusters

if __name__ == "__main__":
    clusters = train_kmeans(TRAIN_PATH)
    pass