"""
    This file aims to convert the Carla's MDLS dataset 
    into 
    kitti format
    References: 
    https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
"""
import src_py.mdlidar_pb2 as reader
from pathlib import Path
import random
import numpy as np
import os

np.random.seed(77)
random.seed(77)

config = {
    'splits': {
        'train': 0.7,
        'test': 0.15,
        'val': 0.15
    }
}

# Creating file structure
Path("../as_kitti").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/ImageSets").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/training/calib").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/training/velodyne").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/training/label_2").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/training/image_2").mkdir(parents=True, exist_ok=True)

Path("../as_kitti/testing/calib").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/testing/velodyne").mkdir(parents=True, exist_ok=True)
Path("../as_kitti/testing/image_2").mkdir(parents=True, exist_ok=True)

towns = ['town_1', 'town_2']
num_files_in_towns = [None] * len(towns)
# Create ImageSets
num_files = 0
for i in range(len(towns)):
    town = towns[i]
    path, dirs, files = next(os.walk("../" + town + '/frame'))
    num_files_in_towns[i] = len(files)
    num_files += len(files)

# Splitting the towns data into train, test, val
files = np.arange(0, num_files)
np.random.shuffle(files)

train_split_ind = int(config['splits']['train'] * num_files)
test_split_ind = train_split_ind + int(config['splits']['test'] * num_files)
train_splits = files[0: train_split_ind]
test_splits = files[train_split_ind: test_split_ind]
val_splits = files[test_split_ind:]

np.savetxt('../as_kitti/ImageSets/train.txt', train_splits, fmt='%04d')
np.savetxt('../as_kitti/ImageSets/test.txt', test_splits, fmt='%04d')
np.savetxt('../as_kitti/ImageSets/val.txt', val_splits, fmt='%04d')

