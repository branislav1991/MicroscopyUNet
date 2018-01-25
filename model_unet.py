import numpy as np
import tensorflow as tf

from common import IoU

# this is where checkpoints from this model will be saved
CHECKPOINT_DIR = "./checkpoints/unet"

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def define_model():
    X = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
    Y_ = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 1])
    lr = tf.placeholder(tf.float32)

    net = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.relu, padding='same')

    net = tf.layers.dropout(net, rate=0.2)

    net = tf.layers.conv2d(X, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d(net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d(net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')

    net = tf.layers.dropout(net, rate=0.2)
    tf.summary.histogram("hidden_hist", net)

    net = tf.layers.conv2d_transpose(net, 64, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d_transpose(net, 64, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d_transpose(net, 64, [3, 3], activation=tf.nn.relu, padding='same')

    net = tf.layers.dropout(net, rate=0.2)

    net = tf.layers.conv2d_transpose(X, 32, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d_transpose(net, 32, [3, 3], activation=tf.nn.relu, padding='same')
    net = tf.layers.conv2d_transpose(net, 32, [3, 3], activation=tf.nn.relu, padding='same')

    logits = tf.layers.conv2d_transpose(net, 1, [3, 3], padding='same')

    #logits = tf.layers.conv2d_transpose(net, 64, [3, 3], dilation_rate=(2, 2), activation='relu')
    #logits = deconv2d(net, 1, 128, 1, 27, "logits_deconv")

    return X, Y_, logits, lr