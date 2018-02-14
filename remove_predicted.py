import os
import shutil

train_path = "./data/stage1_train_small/"
test_path = "./data/stage1_test/"

def remove_predicted_folders(ids):
    for pathar in ids:
        if os.path.exists(pathar[0] + "/masks_predicted"):
            shutil.rmtree(pathar[0] + "/masks_predicted")

train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]
remove_predicted_folders(train_ids)

test_ids = next(os.walk(test_path))
test_ids = [[test_ids[0] + d,d] for d in test_ids[1]]
remove_predicted_folders(test_ids)
