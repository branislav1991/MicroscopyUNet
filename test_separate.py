import os
from kmeans import test_kmeans 
from test_mask_rcnn import test_mask_rcnn

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "mask_rcnn")

TRAIN_PATH = ".\\data\\stage1_train_small\\"
TEST_PATH = ".\\data\\stage1_test\\"

# training dataset
print("Separating training datasets using kmeans")
clusters = test_kmeans(TRAIN_PATH)

print("Testing training clusters")
for i, cluster_ in enumerate(clusters):
    print("Training cluster {0}".format(i))

    # train mask RCNN for every separate cluster
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, "cluster_{0}".format(i))

    test_mask_rcnn(cluster_, TRAIN_PATH, checkpoint_dir)

# testing dataset
print("Separating testing datasets using kmeans")
clusters = test_kmeans(TEST_PATH)

print("Testing testing clusters")
for i, cluster_ in enumerate(clusters):
    print("Testing cluster {0}".format(i))

    # train mask RCNN for every separate cluster
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, "cluster_{0}".format(i))

    test_mask_rcnn(cluster_, TEST_PATH, checkpoint_dir)