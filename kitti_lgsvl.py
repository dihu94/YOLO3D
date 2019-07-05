"""
    Defines class to load the KITTI dataset. 
"""


from __future__ import division
import os
import os.path
import torch
import numpy as np
import cv2
import math
import json 

from utils import *


# Load json configs 
with open('config.json', 'r') as f:
    config = json.load(f)
boundary = config["boundary"]


class KittiDataset(torch.utils.data.Dataset):

    def __init__(self, root = '/home/di/YOLO3D/lgsvl_data/', set = 'train', type = 'velodyne_train'):
        
        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")
        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

    def __getitem__(self, i):
        
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.npy'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'
        
        if self.type == 'velodyne_train':
            calib = load_kitti_calib(calib_file)  
            target = get_target(label_file,calib['Tr_velo2cam'])
        
            # load point cloud data
            a = np.load(lidar_file) #when load synthetized lidar data
            #
            # print("Orig lidar")
            zero_buffer = np.ones((len(a),4)) #used to format synthetized lidar data
            zero_buffer[0:,:-1] = a #used to format synthetized lidar data
            a = zero_buffer #used to format synthetized lidar data

            b = removePoints(a, boundary)
            data = makeBVFeature(b, boundary, 40 / 512)
            return data , target

        elif self.type == 'velodyne_test':
            NotImplemented
        
        else:
            raise ValueError('the type invalid')

    def __len__(self):
        return len(self.file_list)
