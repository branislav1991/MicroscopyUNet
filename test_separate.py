import os
from kmeans import test_kmeans 
from test_mask_rcnn import 

TRAIN_PATH = ".\\data\\stage1_train\\"
VAL_PATH = ".\\data\\stage1_val\\"

LEARNING_RATE = 0.001
 
val_ids = next(os.walk(VAL_PATH))
val_ids = [[val_ids[0] + d,d] for d in val_ids[1]]

print("Separating datasets using kmeans")
clusters = train_kmeans(TRAIN_PATH, load_fitted=False)
clusters = test