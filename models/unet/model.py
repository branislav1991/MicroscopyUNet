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
            self.num_epochs = 300

        if 'display_rate' in kwargs:
            self.display_rate = kwargs['display_rate']
        else:
            self.display_rate = 0

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 1e-4

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 4

        if 'segmentation_thres' in kwargs:
            self.segmentation_thres = kwargs['segmentation_thres']
        else:
            self.segmentation_thres = 0.5

class UNetTestConfig():
    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 16

class Model():
    # this is where checkpoints from this model will be saved
    CHECKPOINT_DIR = "./checkpoints/unet"

    IMG_WIDTH = 192
    IMG_HEIGHT = 192
    IMG_CHANNELS = 1 # we try with only gray or L value

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, Model.IMG_CHANNELS])
        self.Y_ = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, 1])
        self.lr = tf.placeholder(tf.float32)

        self.conv1 = tf.layers.conv2d(self.X, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv1 shape: {0}".format(self.conv1.shape))
        self.conv1 = tf.layers.conv2d(self.conv1, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv1 shape: {0}".format(self.conv1.shape))
        self.pool1 = tf.layers.max_pooling2d(self.conv1, [2, 2], [2, 2])
        print("pool1 shape: {0}".format(self.pool1.shape))

        self.conv2 = tf.layers.conv2d(self.pool1, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv2 shape: {0}".format(self.conv2.shape))
        self.conv2 = tf.layers.conv2d(self.conv2, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv2 shape: {0}".format(self.conv2.shape))
        self.pool2 = tf.layers.max_pooling2d(self.conv2, [2, 2], [2, 2])
        print("pool2 shape: {0}".format(self.pool2.shape))

        self.conv3 = tf.layers.conv2d(self.pool2, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv3 shape: {0}".format(self.conv3.shape))
        self.conv3 = tf.layers.conv2d(self.conv3, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv3 shape: {0}".format(self.conv3.shape))
        self.pool3 = tf.layers.max_pooling2d(self.conv3, [2, 2], [2, 2])
        print("pool3 shape: {0}".format(self.pool3.shape))

        self.conv4 = tf.layers.conv2d(self.pool3, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv4 shape: {0}".format(self.conv4.shape))
        self.conv4 = tf.layers.conv2d(self.conv4, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv4 shape: {0}".format(self.conv4.shape))
        self.drop4 = tf.layers.dropout(self.conv4, rate=0.3)
        self.pool4 = tf.layers.max_pooling2d(self.drop4, [2, 2], [2, 2])
        print("pool4 shape: {0}".format(self.pool4.shape))

        self.conv5 = tf.layers.conv2d(self.pool4, 1024, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv5 shape: {0}".format(self.conv5.shape))
        self.conv5 = tf.layers.conv2d(self.conv5, 1024, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print("conv5 shape: {0}".format(self.conv5.shape))
        self.drop5 = tf.layers.dropout(self.conv5, rate=0.3)



        self.up6 = tf.image.resize_nearest_neighbor(self.drop5, [self.drop5.shape[1] * 2, self.drop5.shape[2] * 2])
        self.up6 = tf.layers.conv2d(self.up6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.merge6 = tf.concat([self.drop4, self.up6], 3)
        self.conv6 = tf.layers.conv2d(self.merge6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv6 = tf.layers.conv2d(self.conv6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        self.up7 = tf.image.resize_nearest_neighbor(self.conv6, [self.conv6.shape[1] * 2, self.conv6.shape[2] * 2])
        self.up7 = tf.layers.conv2d(self.up7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.merge7 = tf.concat([self.conv3, self.up7], 3)
        self.conv7 = tf.layers.conv2d(self.merge7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv7 = tf.layers.conv2d(self.conv7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        self.up8 = tf.image.resize_nearest_neighbor(self.conv7, [self.conv7.shape[1] * 2, self.conv7.shape[2] * 2])
        self.up8 = tf.layers.conv2d(self.up8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.merge8 = tf.concat([self.conv2, self.up8], 3)
        self.conv8 = tf.layers.conv2d(self.merge8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv8 = tf.layers.conv2d(self.conv8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        self.up9 = tf.image.resize_nearest_neighbor(self.conv8, [self.conv8.shape[1] * 2, self.conv8.shape[2] * 2])
        self.up9 = tf.layers.conv2d(self.up9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.merge9 = tf.concat([self.conv1, self.up9], 3)
        self.conv9 = tf.layers.conv2d(self.merge9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv9 = tf.layers.conv2d(self.conv9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv9 = tf.layers.conv2d(self.conv9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.logits = tf.layers.conv2d(self.conv9, 1, [1, 1], padding='same')
        tf.summary.histogram("logits_hist", self.logits)
        self.Y_p = tf.sigmoid(self.logits)


        # self.net = tf.layers.conv2d(self.X, 32, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')

        # self.net = tf.layers.dropout(self.net, rate=0.2)

        # self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d(self.net, 64, [3, 3], dilation_rate=(2, 2), activation=tf.nn.relu, padding='same')

        # self.net = tf.layers.dropout(self.net, rate=0.2)
        # tf.summary.histogram("hidden_hist", self.net)

        # self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d_transpose(self.net, 64, [3, 3], activation=tf.nn.relu, padding='same')

        # self.net = tf.layers.dropout(self.net, rate=0.2)

        # self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')
        # self.net = tf.layers.conv2d_transpose(self.net, 32, [3, 3], activation=tf.nn.relu, padding='same')

        self.loss = tf.losses.sigmoid_cross_entropy(self.Y_, self.logits)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        self.initializers = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.tbmerge = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, X_train, Y_train, config, X_val=None, Y_val=None):
        Y_train = Y_train.astype(np.float32)

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

                if (config.display_rate != 0) and (i % config.display_rate == 0) and (X_val is not None) and (Y_val is not None):
                    # Evaluation on validation dataset
                    dataset_size_val = X_val.shape[0]
                    num_batches_val = math.ceil(float(dataset_size_val)/config.batch_size)

                    Y_p_value = np.zeros(Y_val.shape, dtype=np.float32)
                    loss_value = np.zeros(num_batches_val, dtype=np.float32) 
                    for j in range(0, num_batches_val):
                        d_start = j*config.batch_size
                        d_end = (j+1)*config.batch_size

                        batch_X_val = X_val[d_start:d_end, ...]
                        batch_Y_val = Y_val[d_start:d_end, ...]

                        feed_dict = {self.X: batch_X_val, self.Y_: batch_Y_val.astype(np.float32)}
                        loss_value[j], Y_p_value[d_start:d_end, ...] = sess.run([self.loss, self.Y_p], feed_dict=feed_dict)

                    # calculate mIoU for predicted segmentation labels
                    Y_p_value = Y_p_value > config.segmentation_thres
                    mIoU_value = mIoU(Y_val, Y_p_value)
                    loss_value = np.mean(loss_value)
                    print("Epoch {0}: Validation loss: {1}, Mean IoU: {2}".format(i, loss_value, mIoU_value))

                # save model checkpoint each epoch
                self.saver.save(sess, Model.CHECKPOINT_DIR + "/unet", global_step=i)

    def test(self, X_test, config, Y_test=None):
        dataset_size = X_test.shape[0]
        num_batches = math.ceil(float(dataset_size)/config.batch_size)

        Y_p_value = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[2],1), dtype=np.float32)
        loss_value = np.zeros(num_batches, dtype=np.float32)

        with tf.Session() as sess:
            checkpoint_path = tf.train.latest_checkpoint(Model.CHECKPOINT_DIR)
            self.saver.restore(sess, checkpoint_path)

            for j in tqdm(range(0, num_batches), total=num_batches):
                d_start = j*config.batch_size
                d_end = (j+1)*config.batch_size

                batch_X = X_test[d_start:d_end, ...]
                feed_dict = {self.X: batch_X}
                Y_p_value[d_start:d_end, ...] = sess.run(self.Y_p, feed_dict=feed_dict)
                loss_value[j] = None

                if Y_test is not None:
                    feed_dict[self.Y_] = Y_test[d_start:d_end, ...]
                    loss_value[j] = sess.run(self.loss, feed_dict=feed_dict)

            if Y_test is not None:
                loss_value = np.mean(loss_value)

        return loss_value, Y_p_value