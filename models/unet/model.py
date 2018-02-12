import numpy as np
import tensorflow as tf

import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label

from common import IoU, mIoU, dice, SEGMENTATION_THRESHOLD
from data_provider import TestDataProvider

class UNetTrainConfig():
    def __init__(self, **kwargs):
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            self.num_epochs = 100

        if 'val_rate' in kwargs:
            self.val_rate = kwargs['val_rate']
        else:
            self.val_rate = 0

        if 'starter_learning_rate' in kwargs:
            self.starter_learning_rate = kwargs['starter_learning_rate']
        else:
            self.starter_learning_rate = 0.01

class Model():
    # this is where checkpoints from this model will be saved
    CHECKPOINT_DIR = "./checkpoints/unet"
    NUM_CLASSES = 1

    def __init__(self, num_img_channels, num_scales=1):
        #self.sess = tf.Session()
        #tf.keras.backend.set_session(self.sess)

        #vgg16 = tf.keras.applications.VGG16(include_top=False, 
        #    input_shape=(Model.IMG_WIDTH,Model.IMG_HEIGHT,Model.IMG_CHANNELS))

        self.X = tf.placeholder(tf.float32, [None, None, None, num_img_channels], name='input')
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, num_img_channels], name='ground_truth')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.increment_global_step_op = tf.assign_add(self.global_step, 1)
        self.starter_learning_rate = tf.placeholder(tf.float32, name='starter_learning_rate')
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, 
            self.global_step, 5, 0.1)

        # build the scales and append them to list
        scales = [self.X]
        for i in range(1, num_scales):
            scales.append(tf.image.resize_bilinear(scales[i-1], [scales[i-1].shape[1] // 2, scales[i-1].shape[2] // 2])) 

        scales_out = []
        for i, scale in enumerate(scales):
            with tf.name_scope('scale_{0}'.format(i)):
                skip_4, skip_3, skip_2, skip_1 = self._build_encoder(scale)
                scales_out.append(self._build_decoder(skip_4, skip_3, skip_2, skip_1))

        # upsample all scales to original scale
        for i in range(1, len(scales_out)):
            scales_out[i] = tf.image.resize_bilinear(scales_out[i], [scales_out[0].shape[1], scales_out[0].shape[2]])

        scale_cat = tf.concat(scales_out, axis=3)

        # fully connected layers for final
        logits = tf.layers.conv2d(scale_cat, 32, [1, 1], padding='same')
        logits = tf.layers.conv2d(logits, Model.NUM_CLASSES, [1, 1], padding='same')
        self.Y_p = tf.nn.sigmoid(logits)

        #self.loss = tf.losses.sigmoid_cross_entropy(self.Y_, logits)
        self.loss = 1-dice(self.Y_, logits)

        self.tb_train_loss = tf.summary.scalar('training_loss', self.loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.initializers = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.saver = tf.train.Saver(max_to_keep=100)

    def _build_encoder(self, X):
        with tf.name_scope('encoder'):
            conv1 = tf.layers.conv2d(X, 32, [3,3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv1 shape: {0}".format(conv1.shape))
            pool1 = tf.layers.max_pooling2d(conv1, [2,2], [2,2], padding='same')
            print("pool1 shape: {0}".format(pool1.shape))

            conv2 = tf.layers.conv2d(pool1, 64, [3,3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv2 shape: {0}".format(conv2.shape))
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2,2], padding='same')
            print("pool2 shape: {0}".format(pool2.shape))

            conv3 = tf.layers.conv2d(pool2, 128, [3,3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv3 shape: {0}".format(conv3.shape))
            pool3 = tf.layers.max_pooling2d(conv3, [2,2], [2,2], padding='same')
            print("pool3 shape: {0}".format(pool3.shape))

            conv4 = tf.layers.conv2d(pool3, 512, [1,1], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv4 shape: {0}".format(conv3.shape))
            drop4 = tf.layers.dropout(conv4, rate=0.5)

        return drop4, conv3, conv2, conv1

    def crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], x1_shape[3]]
        x1_crop = tf.slice(x1, offsets, size)
        merge = tf.concat([x1_crop, x2], 3)
        merge.set_shape([None,None,None,x1.shape[3] + x2.shape[3]])
        return merge

    def _build_decoder(self, skip_4, skip_3, skip_2, skip_1):
        up3 = tf.layers.conv2d_transpose(skip_4, 128, [3,3], padding='same', strides=(2,2),
            activation=tf.nn.relu)
        up3 = up3[:,0:skip_3.shape[1],0:skip_3.shape[2],:]
        #up3 = tf.image.resize_bilinear(skip_4, [skip_4.shape[1] * 2, skip_4.shape[2] * 2])
        #up3 = tf.layers.conv2d(up3, 128, [3, 3], padding='same', activation=tf.nn.relu)
        merge3 = self.crop_and_concat(up3, skip_3)

        up2 = tf.layers.conv2d_transpose(merge3, 64, [3,3], padding='same', strides=(2,2),
            activation=tf.nn.relu)
        up2 = up2[:,0:skip_2.shape[1],0:skip_2.shape[2],:]
        #up2 = tf.image.resize_bilinear(merge3, [merge3.shape[1] * 2, merge3.shape[2] * 2])
        #up2 = tf.layers.conv2d(up2, 64, [3, 3], padding='same')
        merge2 = self.crop_and_concat(up2, skip_2)

        up1 = tf.layers.conv2d_transpose(merge2, 32, [3,3], padding='same', strides=(2,2),
            activation=tf.nn.relu)
        up1 = up1[:,0:skip_1.shape[1],0:skip_1.shape[2],:]
        #up1 = tf.image.resize_bilinear(merge2, [merge2.shape[1] * 2, merge2.shape[2] * 2])
        #up1 = tf.layers.conv2d(up1, 32, [3, 3], padding='same')
        merge1 = self.crop_and_concat(up1, skip_1)
        conv1 = tf.layers.conv2d(merge1, 32, [3, 3], padding='same')

        return conv1

    def train(self, config, data_provider_train, restore=False, data_provider_val=None):
        has_validation = data_provider_val is not None
        num_batches_train = data_provider_train.num_batches()
        num_batches_val = data_provider_val.num_batches()

        with tf.Session() as sess:
            if restore == True:
                checkpoint_path = tf.train.latest_checkpoint(Model.CHECKPOINT_DIR)
                self.saver.restore(sess, checkpoint_path)
            else:
                sess.run(self.initializers)

            train_writer = tf.summary.FileWriter(".tensorboard/unet", sess.graph)

            for i in range(0, config.num_epochs):
                for j, batch in tqdm(enumerate(data_provider_train), total=num_batches_train):
                    feed_dict = {self.X: batch[0], self.Y_: batch[1], 
                        self.starter_learning_rate: config.starter_learning_rate}
                    summary,loss_value,_ = sess.run([self.tb_train_loss, self.loss, self.optimizer], 
                        feed_dict=feed_dict)
                    print("Loss: {0}".format(loss_value))
                    
                    train_writer.add_summary(summary, num_batches_train*i + j)

                if (config.val_rate != 0) and (i % config.val_rate == 0) and (has_validation == True):
                    Y_p_vals = []
                    loss_vals = np.zeros(num_batches_val, dtype=np.float32)

                    for j, batch in enumerate(data_provider_val):
                        feed_dict = {self.X: batch[0], self.Y_: batch[1]}
                        loss_vals[j], Y_p_ = sess.run([self.loss, self.Y_p], feed_dict=feed_dict)
                        Y_p_vals.append(Y_p_)

                    # calculate mIoU for predicted segmentation labels
                    true_Y = data_provider_val.get_true_Y()
                    IoU_value = np.zeros(len(Y_p_vals))
                    for k in range(0, len(Y_p_vals)):
                        p = Y_p_vals[k] > SEGMENTATION_THRESHOLD
                        IoU_value[k] = IoU(true_Y[k][None,...], p)
                    
                    mIoU_value = np.mean(IoU_value)
                    loss_value = np.mean(loss_vals)

                    # tensorboard
                    summary = tf.Summary()
                    summary.value.add(tag='validation_loss', simple_value=loss_value)
                    summary.value.add(tag='validation_IoU', simple_value=mIoU_value)
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    print("Epoch {0}: Validation loss: {1}, Mean IoU: {2}".format(i, loss_value, mIoU_value))

                    data_provider_val.reset()

                # save model checkpoint each epoch
                self.saver.save(sess, Model.CHECKPOINT_DIR + "/unet", global_step=i)
                data_provider_train.reset()

                # increment global step
                sess.run(self.increment_global_step_op)

    def test(self, data_provider):
        is_labeled_data = not isinstance(data_provider, TestDataProvider)
        num_batches = data_provider.num_batches()

        Y_p_vals = []
        loss_vals = np.zeros(num_batches, dtype=np.float32)
        loss_value = None

        with tf.Session() as sess:
            checkpoint_path = tf.train.latest_checkpoint(Model.CHECKPOINT_DIR)
            self.saver.restore(sess, checkpoint_path)

            for j, batch in tqdm(enumerate(data_provider), total=num_batches):
                if not is_labeled_data:
                    feed_dict = {self.X: batch}
                    Y_p_ = sess.run(self.Y_p, feed_dict=feed_dict)
                    Y_p_vals.append(Y_p_)
                    loss_vals[j] = None

                else:
                    feed_dict = {self.X: batch[0], self.Y_: batch[1]}
                    loss_vals[j], Y_p_ = sess.run([self.loss, self.Y_p], feed_dict=feed_dict)
                    Y_p_vals.append(Y_p_)

            if is_labeled_data:
               loss_value = np.mean(loss_vals)

        return loss_value, Y_p_vals