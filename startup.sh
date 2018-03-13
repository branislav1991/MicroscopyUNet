#! /bin/bash
# Description: Kaggle Data Science Bowl 18 Challenge. 
# This is the startup script that is called by the Google
# Compute Instance launcher script.
# Author: Branislav Hollaender
# Date: 03/11/2018

## delete previous instance_info.txt
sudo rm instance_info.txt

## delete previous checkpoints
sudo rm -r checkpoints/mask_rcnn

## get current version of script on github
git pull >> instance_info.txt

## run python script
nohup sudo ~/.pyenv/versions/MLPlayground/bin/python train_mask_rcnn.py "./data/stage1_simple/" "./data/stage1_val/" >> instance_info.txt &

## Shutdown instance. Note: this just shuts down the instance-not delete it.
## sudo shutdown -h now