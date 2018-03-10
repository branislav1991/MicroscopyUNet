import os
from sys import float_info
from hyperopt import fmin, tpe, hp
from train_mask_rcnn import train_mask_rcnn, CHECKPOINT_DIR
import pickle
import math
from keras import backend as K

train_path="./data/stage1_train/"
val_path="./data/stage1_val/"

train_ids = next(os.walk(train_path))
train_ids = [[train_ids[0] + d,d] for d in train_ids[1]]

val_ids = next(os.walk(val_path))
val_ids = [[val_ids[0] + d,d] for d in val_ids[1]]

def objective(args):
    lr_heads = args["lr_heads"]
    lr_all = args["lr_all"]
    histories = train_mask_rcnn(train_ids, val_ids, init_with="coco", checkpoint_dir=CHECKPOINT_DIR,
            procedures=[{"layers": "heads", "learning_rate": lr_heads, "epochs": 5},
                        {"layers": "all", "learning_rate": lr_all, "epochs": 5}])
    if "val_loss" in histories[0].history:
        h = histories[0].history["val_loss"][-1].flat[0]
        if math.isfinite(h) is not True:
            h = float_info.max
    else:
        h = float_info.max
    K.clear_session()
    return h

space = {"lr_heads": hp.loguniform('lr_heads', math.log(0.0001), math.log(0.1)),
         "lr_all": hp.loguniform('lr_all', math.log(0.0001), math.log(0.1))} 

best = fmin(objective, space, algo=tpe.suggest, max_evals=15)
print(best)

# save best hyperparameters to file
