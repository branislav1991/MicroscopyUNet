import os
from kmeans import train_kmeans
from train_mask_rcnn import train_mask_rcnn, CHECKPOINT_DIR, create_folder

TRAIN_PATH = ".\\data\\stage1_train\\"
VAL_PATH = ".\\data\\stage1_val\\"

LEARNING_RATE = 0.001
 
val_ids = next(os.walk(VAL_PATH))
val_ids = [[val_ids[0] + d,d] for d in val_ids[1]]

print("Separating datasets using kmeans")
clusters = train_kmeans(TRAIN_PATH, load_fitted=False)

for i, cluster_ in enumerate(clusters):
    print("Training cluster {0}".format(i))

    # train mask RCNN for every separate cluster
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, "cluster_{0}".format(i))
    create_folder(checkpoint_dir)
    
    train_mask_rcnn(cluster_, val_ids, init_with="coco", checkpoint_dir=checkpoint_dir,
        procedures=[{"layers": "heads", "learning_rate": LEARNING_RATE, "epochs": 5}, 
                    {"layers": "5+", "learning_rate": LEARNING_RATE/10, "epochs": 10}])