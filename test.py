import numpy as np
import tensorflow as tf
import math

from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.morphology import label
import scipy

from model_unet import define_model, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, CHECKPOINT_DIR
from common import mIoU, create_folder, create_predicted_folders, load_train_images, load_test_images

# Set some parameters
#TRAIN_PATH = './data/stage1_train/'
TRAIN_PATH = './data/stage1_train_small/'
TEST_PATH = './data/stage1_test/'
SEGMENTATION_THRESHOLD = 0.5

print('Getting and resizing train images and masks ... ')
X_train, Y_train, sizes_train, train_ids = load_train_images(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
X_test, sizes_test, test_ids = load_test_images(TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
print('Done loading images!')

create_predicted_folders(train_ids)
create_predicted_folders(test_ids)

X, Y_, logits, lr = define_model()

Y_p = tf.sigmoid(logits)

loss = tf.losses.sigmoid_cross_entropy(Y_, logits)
saver = tf.train.Saver()

with tf.Session() as sess:
    checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    saver.restore(sess, checkpoint_path)

    print("Beginning evaluation ...")
    print("Evaluating training data ...")
    feed_dict = {X: X_train, Y_: Y_train}
    loss_value, Y_p_value = sess.run([loss, Y_p], feed_dict=feed_dict)

    Y_p_value = Y_p_value > SEGMENTATION_THRESHOLD
    mIoU_value = mIoU(Y_train, Y_p_value)
    print("Loss: {0}, Mean IoU: {1}".format(loss_value, mIoU_value))

    for n in tqdm(range(0, Y_p_value.shape[0]), total=Y_p_value.shape[0]):
        path = train_ids[n][0] + "/masks_predicted"
        mask_resized = scipy.misc.imresize(Y_p_value[n,:,:,0], sizes_train[n], interp='nearest')
        io.imsave(path + "/mask.tif", mask_resized)

    print("Evaluating test data ...")
    feed_dict = {X: X_test}
    Y_p_value = sess.run(Y_p, feed_dict=feed_dict)
    Y_p_value = Y_p_value > SEGMENTATION_THRESHOLD
    Y_p_value = (Y_p_value).astype(int)

    for n in tqdm(range(0, Y_p_value.shape[0]), total=Y_p_value.shape[0]):
        path = test_ids[n][0] + "/masks_predicted"
        mask_resized = scipy.misc.imresize(Y_p_value[n,:,:,0], sizes_test[n], interp='nearest')
        io.imsave(path + "/mask.tif", mask_resized)

    print("Done evaluation!")