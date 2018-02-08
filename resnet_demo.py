import numpy as np
import tensorflow as tf

model = tf.keras.applications.ResNet50()

checkpoint_path = tf.train.latest_checkpoint('./checkpoints/resnet50')

tf.train.import_meta_graph(checkpoint_path)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)