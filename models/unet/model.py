import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label

from common import IoU, mIoU
from data_provider import TestDataProvider, TrainDataProviderResize

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

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 1e-4

        if 'segmentation_thres' in kwargs:
            self.segmentation_thres = kwargs['segmentation_thres']
        else:
            self.segmentation_thres = 0.5

class Model():
    # this is where checkpoints from this model will be saved
    CHECKPOINT_DIR = "./checkpoints/unet"

    IMG_WIDTH = 480
    IMG_HEIGHT = 480
    IMG_CHANNELS = 1 # we try with only gray or L value

    def __init__(self, num_scales = 1):
        self.X = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, Model.IMG_CHANNELS], name='input')
        self.Y_ = tf.placeholder(tf.float32, [None, Model.IMG_WIDTH, Model.IMG_HEIGHT, 1], name='ground_truth')
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # build the scales and append them to list
        scales = [self.X]
        for i in range(1, num_scales):
            scales.append(tf.image.resize_bilinear(scales[i-1], [scales[i-1].shape[1] // 2, scales[i-1].shape[2] // 2])) 

        scales_out = []
        for i, scale in enumerate(scales):
            with tf.name_scope('scale_{0}'.format(i)):
                skip_5, skip_4, skip_3, skip_2, skip_1 = self._build_encoder(scale)
                scales_out.append(self._build_decoder(skip_5, skip_4, skip_3, skip_2, skip_1))

        # upsample all scales to original scale
        for i in range(1, len(scales_out)):
            scales_out[i] = tf.image.resize_bilinear(scales_out[i], [scales_out[0].shape[1], scales_out[0].shape[2]])

        scale_cat = tf.concat(scales_out, axis=3)

        # fully connected layers for final
        logits = tf.layers.conv2d(scale_cat, 128, [1, 1], padding='same')
        logits = tf.layers.conv2d(logits, 1, [1, 1], padding='same')

        self.Y_p = tf.sigmoid(logits)

        self.loss = tf.losses.sigmoid_cross_entropy(self.Y_, logits)
        self.tb_train_loss = tf.summary.scalar('training_loss', self.loss)

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        self.initializers = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.saver = tf.train.Saver(max_to_keep=100)

    def _build_encoder(self, X):
        with tf.name_scope('encoder'):
            conv1 = tf.layers.conv2d(X, 64, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv1 shape: {0}".format(conv1.shape))
            conv1 = tf.layers.conv2d(conv1, 64, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv1 shape: {0}".format(conv1.shape))
            pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
            print("pool1 shape: {0}".format(pool1.shape))

            conv2 = tf.layers.conv2d(pool1, 128, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv2 shape: {0}".format(conv2.shape))
            conv2 = tf.layers.conv2d(conv2, 128, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv2 shape: {0}".format(conv2.shape))
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])
            print("pool2 shape: {0}".format(pool2.shape))

            conv3 = tf.layers.conv2d(pool2, 256, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv3 shape: {0}".format(conv3.shape))
            conv3 = tf.layers.conv2d(conv3, 256, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv3 shape: {0}".format(conv3.shape))
            pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2])
            print("pool3 shape: {0}".format(pool3.shape))

            conv4 = tf.layers.conv2d(pool3, 512, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv4 shape: {0}".format(conv4.shape))
            conv4 = tf.layers.conv2d(conv4, 512, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv4 shape: {0}".format(conv4.shape))
            drop4 = tf.layers.dropout(conv4, rate=0.5)
            pool4 = tf.layers.max_pooling2d(drop4, [2, 2], [2, 2])
            print("pool4 shape: {0}".format(pool4.shape))

            conv5 = tf.layers.conv2d(pool4, 1024, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv5 shape: {0}".format(conv5.shape))
            conv5 = tf.layers.conv2d(conv5, 1024, [3, 3], activation=tf.nn.relu, padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            print("conv5 shape: {0}".format(conv5.shape))
            drop5 = tf.layers.dropout(conv5, rate=0.5)

        return drop5, drop4, conv3, conv2, conv1

    def _build_decoder(self, skip_5, skip_4, skip_3, skip_2, skip_1):
        up6 = tf.image.resize_nearest_neighbor(skip_5, [skip_5.shape[1] * 2, skip_5.shape[2] * 2])
        up6 = tf.layers.conv2d(up6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge6 = tf.concat([skip_4, up6], 3)
        conv6 = tf.layers.conv2d(merge6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv6 = tf.layers.conv2d(conv6, 512, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        up7 = tf.image.resize_nearest_neighbor(conv6, [conv6.shape[1] * 2, conv6.shape[2] * 2])
        up7 = tf.layers.conv2d(up7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge7 = tf.concat([skip_3, up7], 3)
        conv7 = tf.layers.conv2d(merge7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7 = tf.layers.conv2d(conv7, 256, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        up8 = tf.image.resize_nearest_neighbor(conv7, [conv7.shape[1] * 2, conv7.shape[2] * 2])
        up8 = tf.layers.conv2d(up8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge8 = tf.concat([skip_2, up8], 3)
        conv8 = tf.layers.conv2d(merge8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv8 = tf.layers.conv2d(conv8, 128, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        up9 = tf.image.resize_nearest_neighbor(conv8, [conv8.shape[1] * 2, conv8.shape[2] * 2])
        up9 = tf.layers.conv2d(up9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge9 = tf.concat([skip_1, up9], 3)
        conv9 = tf.layers.conv2d(merge9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv9 = tf.layers.conv2d(conv9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv9 = tf.layers.conv2d(conv9, 64, [3, 3], activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        return conv9

    def train_old(self, X_train, Y_train, config, X_val=None, Y_val=None):
        Y_train = Y_train.astype(np.float32)

        # this is a powerful new idea
        # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

        batch_size = 4

        with tf.Session() as sess:
            
            sess.run(self.initializers)

            train_writer = tf.summary.FileWriter(".tensorboard/unet", sess.graph)

            dataset_size = X_train.shape[0]
            num_batches = math.ceil(float(dataset_size)/batch_size)

            for i in range(0, config.num_epochs):
                for j in tqdm(range(0, num_batches), total=num_batches):
                    d_start = j*batch_size
                    d_end = (j+1)*batch_size

                    batch_X = X_train[d_start:d_end, ...]
                    batch_Y = Y_train[d_start:d_end, ...]
                    feed_dict = {self.X: batch_X, self.Y_: batch_Y, self.lr: config.learning_rate}
                    summary,loss_value,_ = sess.run([self.tb_train_loss, self.loss, self.optimizer], 
                        feed_dict=feed_dict)
                    print("Loss: {0}".format(loss_value))
                    
                    train_writer.add_summary(summary, num_batches*i + j)

                if (config.val_rate != 0) and (i % config.val_rate == 0) and (X_val is not None) and (Y_val is not None):
                    # Evaluation on validation dataset
                    dataset_size_val = X_val.shape[0]
                    num_batches_val = math.ceil(float(dataset_size_val)/batch_size)

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
                    mIoU_value = mIoU(Y_val > 0, Y_p_value)
                    loss_value = np.mean(loss_value)

                    # tensorboard
                    summary = tf.Summary()
                    summary.value.add(tag='validation_loss', simple_value=loss_value)
                    summary.value.add(tag='validation_IoU', simple_value=mIoU_value)
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    print("Epoch {0}: Validation loss: {1}, Mean IoU: {2}".format(i, loss_value, mIoU_value))

                # save model checkpoint each epoch
                self.saver.save(sess, Model.CHECKPOINT_DIR + "/unet", global_step=i)

    def train(self, config, data_provider_train, data_provider_val=None):
        Y_train = Y_train.astype(np.float32)

        # this is a powerful new idea
        # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

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
                    summary,loss_value,_ = sess.run([self.tb_train_loss, self.loss, self.optimizer], 
                        feed_dict=feed_dict)
                    print("Loss: {0}".format(loss_value))
                    
                    train_writer.add_summary(summary, num_batches*i + j)

                if (config.val_rate != 0) and (i % config.val_rate == 0) and (X_val is not None) and (Y_val is not None):
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
                    mIoU_value = mIoU(Y_val > 0, Y_p_value)
                    loss_value = np.mean(loss_value)

                    # tensorboard
                    summary = tf.Summary()
                    summary.value.add(tag='validation_loss', simple_value=loss_value)
                    summary.value.add(tag='validation_IoU', simple_value=mIoU_value)
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    print("Epoch {0}: Validation loss: {1}, Mean IoU: {2}".format(i, loss_value, mIoU_value))

                # save model checkpoint each epoch
                self.saver.save(sess, Model.CHECKPOINT_DIR + "/unet", global_step=i)

    def test(self, data_provider):
        is_labeled_data = not isinstance(data_provider, TestDataProvider)
        batch_size = data_provider.num_batches()

        Y_p_value = []
        loss_value = np.zeros(batch_size, dtype=np.float32)

        with tf.Session() as sess:
            checkpoint_path = tf.train.latest_checkpoint(Model.CHECKPOINT_DIR)
            self.saver.restore(sess, checkpoint_path)

            for j, data in tqdm(enumerate(data_provider), total=batch_size):
                if not is_labeled_data:
                    feed_dict = {self.X: data}
                    y_p = sess.run(self.Y_p, feed_dict=feed_dict)
                    Y_p_value.append(y_p)
                    loss_value[j] = None

                else:
                    feed_dict = {self.X: data[0], self.Y_: data[1]}
                    loss_value[j], y_p = sess.run([self.loss, self.Y_p], feed_dict=feed_dict)
                    Y_p_value.append(y_p)

            if is_labeled_data:
               loss_value = np.mean(loss_value)

        return loss_value, Y_p_value