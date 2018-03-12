#! /bin/bash
# Description: Kaggle Data Science Bowl 18 Challenge. 
# This is the startup script that is called by the Google
# Compute Instance launcher script.
# Author: Branislav Hollaender
# Date: 03/11/2018

## wait for 30 seconds until everything loads
## sleep 30

## behave like superuser
sudo apt-get update

## change to workspace directory
cd ~
cd /home/branislav_hollander/workspace/MicroscopyUNet/
echo "Current working directory:" >> instance_info.txt
echo $PWD >> instance_info.txt

## change pyenv to MLPlayground
pyenv activate MLPlayground
echo "Activated MLPlayground" >> instance_info.txt

## get current version of script on github
git pull >> instance_info.txt

## run python script
sudo /home/branislav_hollander/.pyenv/versions/MLPlayground/bin/python train_mask_rcnn.py >> instance_info.txt

## Shutdown instance. Note: this just shuts down the instance-not delete it.
## sudo shutdown -h now