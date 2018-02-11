import numpy as np # linear algebra
import os
import cv2
import pandas as pd

from tqdm import tqdm
from skimage.morphology import label

TEST_PATH = './data/stage1_test/'

# Any results you write to the current directory are saved as output.
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def main():
    print('Beginning RLE encoding ...')
    test_ids = next(os.walk(TEST_PATH))
    test_ids = [[test_ids[0] + d + '/masks_predicted', d] for d in test_ids[1]]

    new_test_ids = []
    rles = []
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        rle = list(prob_to_rles(cv2.imread(id_[0] + '/mask_inner.tif',cv2.IMREAD_GRAYSCALE)))
        new_test_ids.extend([id_[1]] * len(rle))
        rles.extend(rle)

    submission = pd.DataFrame()
    submission['ImageId'] = new_test_ids
    submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    
    submission.to_csv(TEST_PATH + 'submission.csv', index=False)

    print('Done RLE encoding!')

if __name__ == "__main__":
    main()
