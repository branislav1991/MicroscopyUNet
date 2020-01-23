# Microscopy U-Net
## Description
Source code for the kaggle Data Science Bowl 18 competition. Includes a Mask-RCNN (from Matterport) and a U-Net model.

## Data sources
Data can be obtained at https://www.kaggle.com/c/data-science-bowl-2018/data. Additional data source used is the TNBC dataset (https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html). This dataset must be processed with https://www.kaggle.com/branislav1991/converting-tnbc-external-data-to-dsb2018-format/comments before use.

## How to run the code
Run the testing by starting test_mask_rcnn.py (Python 3.6, Tensorflow 1.6). Predicted masks will be saved in data/stage*_test/[img]/masks_predicted.
