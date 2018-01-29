import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label

from common import IoU, mIoU

class UNetTrainConfig():
    def __init__(self, **kwargs):
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            self.num_epochs = 100

        if 'display_rate' in kwargs:
            self.display_rate = kwargs['display_rate']
        else:
            self.display_rate = 0

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 0.001

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 16

        if 'segmentation_thres' in kwargs:
            self.segmentation_thres = kwargs['segmentation_thres']
        else:
            self.segmentation_thres = 0.5

class UNetTestConfig():
    def __init__(self, **kwargs):
        pass

class Model():
    # this is where checkpoints from this model will be saved
    CHECKPOINT_DIR = "./checkpoints/unet"

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, Model.IMG_CHANNELS])
        self.Y_ = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, 1])
        self.lr = tf.placeholder(tf.float32)

        self.net = tf.layers.conv2d(self.X, 32, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')

        self.net = tf.layers.dropout(self.net, rate=0.2)

        self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')

        self.net = tf.layers.dropout(self.net, rate=0.2)
        tf.summary.histogram("hidden_hist", self.net)

        self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')

        self.net = tf.layers.dropout(self.net, rate=0.2)

        self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')

        self.logits = tf.layers.conv2d_transpose(self.net, 1, [3, 3], padding='same')
        self.Y_p = tf.sigmoid(self.logits)

        self.loss = tf.losses.sigmoid_cross_entropy(self.Y_, self.logits)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.initializers = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.tbmerge = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, X_train, Y_train, config):
        with tf.Session() as sess:
            
            sess.run(self.initializers)

            train_writer = tf.summary.FileWriter(".tensorboard/unet", sess.graph)

            dataset_size = X_train.shape[0]
            num_batches = math.ceil(float(dataset_size)/config.batch_size)

            for i in range(0, config.num_epochs):
                for j in tqdm(range(0, num_batches), total=num_batches):
                    d_start = j*config.batch_size
                    d_end = (j+1)*config.batch_size

                    batch_X = X_train[d_start:d_end, ...]
                    batch_Y = Y_train[d_start:d_end, ...]
                    feed_dict = {self.X: batch_X, self.Y_: batch_Y, self.lr: config.learning_rate}
                    summary,loss_value,_ = sess.run([self.tbmerge, self.loss, self.optimizer], 
                        feed_dict=feed_dict)
                    print("Loss: {0}".format(loss_value))
                    
                    train_writer.add_summary(summary, num_batches*i + j)

                if (config.display_rate != 0) and (i % config.display_rate == 0):
                    # Evaluation on training set (we cannot evaluate on test set since we do not have labels)
                    feed_dict = {self.X: X_train, self.Y_: Y_train}
                    loss_value, Y_p_value = sess.run([self.loss, self.Y_p], feed_dict=feed_dict)

                    # calculate mIoU for predicted segmentation labels
                    Y_p_value = Y_p_value > config.segmentation_thres
                    mIoU_value = mIoU(Y_train, Y_p_value)
                    print("Epoch {0}: Loss: {1}, Mean IoU: {2}".format(i, loss_value, mIoU_value))

                # save model checkpoint each epoch
                self.saver.save(sess, Model.CHECKPOINT_DIR + "/unet", global_step=i)

    def test(self, X_test, config, Y_test=None):
        with tf.Session() as sess:
            checkpoint_path = tf.train.latest_checkpoint(Model.CHECKPOINT_DIR)
            self.saver.restore(sess, checkpoint_path)

            feed_dict = {self.X: X_test}
            Y_p_value = sess.run(self.Y_p, feed_dict=feed_dict)
            loss_value = None

            if Y_test is not None:
                feed_dict[self.Y_] = Y_test
                loss_value = sess.run(self.loss, feed_dict=feed_dict)

        return loss_value, Y_p_value