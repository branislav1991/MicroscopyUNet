import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
import pickle

IMG_WIDTH = 256
IMG_HEIGHT = 256

KMEANS_PATH = ".\\checkpoints\\kmeans.pkl"

def test_kmeans(test_path):
    with open(KMEANS_PATH, 'rb') as file:
        kmeans = pickle.load(file)

    test_ids = next(os.walk(test_path))
    test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
    test_id_images = [d[0] + '\\images\\' + d[1] + ".png" for d in test_ids]

    imgs = np.zeros((len(test_id_images), IMG_HEIGHT*IMG_WIDTH*3))
    for i, id_ in enumerate(test_id_images):
        img_ = imread(id_)[:,:,:3]
        img_ = resize(img_, (IMG_HEIGHT, IMG_WIDTH))
        imgs[i,:] = img_.flatten()

    labels = kmeans.predict(imgs)

    clustered = zip(test_ids, labels)
    clustered = sorted(clustered, key=lambda x: x[1])

    clusters = []
    clusters.append([x[0] for x in clustered if x[1]==0])
    clusters.append([x[0] for x in clustered if x[1]==1])
    clusters.append([x[0] for x in clustered if x[1]==2])

    return clusters

def train_kmeans(train_path, load_fitted=False):
    train_ids = next(os.walk(train_path))
    train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
    train_id_images = [d[0] + '\\images\\' + d[1] + ".png" for d in train_ids]
    
    imgs = np.zeros((len(train_id_images), IMG_HEIGHT*IMG_WIDTH*3))
    for i, id_ in enumerate(train_id_images):
        img_ = imread(id_)[:,:,:3]
        img_ = resize(img_, (IMG_HEIGHT, IMG_WIDTH))
        imgs[i,:] = img_.flatten()

    if load_fitted==False:
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(imgs)
        
        # save fitted k-means
        with open(KMEANS_PATH, 'wb') as file:
            pickle.dump(kmeans, file)
    else:
        with open(KMEANS_PATH, 'rb') as file:
            kmeans = pickle.load(file)
    
    labels = kmeans.labels_

    clustered = zip(train_ids, labels)
    clustered = sorted(clustered, key=lambda x: x[1])

    clusters = []
    clusters.append([x[0] for x in clustered if x[1]==0])
    clusters.append([x[0] for x in clustered if x[1]==1])
    clusters.append([x[0] for x in clustered if x[1]==2])
    return clusters