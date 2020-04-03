import cv2
import os
import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image

keypoint = np.load('keypoints.npy')

import pdb; pdb.set_trace()

generator_output_size = 256
half_side_length = 350
side_lenth = half_side_length * 2
x_center = 960
y_center = 400

x_offset = x_center - half_side_length
y_offset = y_center - half_side_length

for i in keypoint.shape[0]:
    np.savetxt('deployment/vid2vid/datasets/face/test_keypoints/0001/keypoints_{:05d}.txt'.format(i), keypoint[i].astype(np.uint8) / generator_output_size * side_lenth + [x_offset,y_offset] , fmt='%3i', delimiter=',')