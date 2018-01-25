import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label

from model_unet import define_model, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, CHECKPOINT_DIR
from common import mIoU, create_folder, load_train_images

# create checkpoint folder
create_folder(CHECKPOINT_DIR)

#TRAIN_PATH = './data/stage1_train/'
TRAIN_PATH = './data/stage1_train_small/'
NUM_EPOCHS = 200
DISPLAY_RATE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16
SEGMENTATION_THRESHOLD = 0.5

print('Getting and resizing train images and masks ... ')
X_train, Y_train, sizes_train, _ = load_train_images(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
print('Done loading images!')

X, Y_, logits, lr = define_model()
Y_p = tf.sigmoid(logits)

loss = tf.losses.sigmoid_cross_entropy(Y_, logits)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
initializers = [tf.global_variables_initializer(), tf.local_variables_initializer()]
tbmerge = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    print('Beginning training ... ')
    sess.run(initializers)

    train_writer = tf.summary.FileWriter(".tensorboard/unet", sess.graph)

    dataset_size = X_train.shape[0]
    num_batches = math.ceil(float(dataset_size)/BATCH_SIZE)

    for i in range(0, NUM_EPOCHS):
        for j in tqdm(range(0, num_batches), total=num_batches):
            batch_X = X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE, ...]
            batch_Y = Y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE, ...]
            feed_dict = {X: batch_X, Y_: batch_Y, lr: LEARNING_RATE}
            summary,loss_value,_ = sess.run([tbmerge, loss, optimizer], 
                feed_dict=feed_dict)
            print("Loss: {0}".format(loss_value))
            
            train_writer.add_summary(summary, num_batches*i + j)

        # if i % DISPLAY_RATE == 0:
        #     # Evaluation on training set (we cannot evaluate on test set since we do not have labels)
        #     feed_dict = {X: X_train, Y_: Y_train, lr: LEARNING_RATE}
        #     loss_value, Y_p_value = sess.run([loss, Y_p], feed_dict=feed_dict)

        #     # calculate mIoU for predicted segmentation labels
        #     Y_p_value = Y_p_value > SEGMENTATION_THRESHOLD
        #     mIoU_value = mIoU(Y_train, Y_p_value)
        #     print("Epoch {0}: Loss: {1}, Mean IoU: {2}".format(i,loss_value, mIoU_value))

        # save model checkpoint each epoch
        saver.save(sess, CHECKPOINT_DIR + "/unet", global_step=i)

    print("Done training!")