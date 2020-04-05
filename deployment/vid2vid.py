import cv2
import os
import subprocess
import numpy as np
import torch
import shutil
from torchvision.utils import save_image
from PIL import Image

keypoint = np.load('deployment/local_input/keypoints.npy')
source_image_dir = 'deployment/local_input/0001'

generator_output_size = 256
half_side_length = 350
side_lenth = half_side_length * 2
x_center = 960
y_center = 400

x_offset = x_center - half_side_length
y_offset = y_center - half_side_length

if not os.path.exists('deployment/vid2vid/datasets/'):
	os.system("deployment/setup_vid2vid.sh")

landmark_dir = 'deployment/vid2vid/datasets/face/test_keypoints/0001/'
shutil.rmtree(landmark_dir)
os.mkdir(landmark_dir)
for i in range(keypoint.shape[0]):
    np.savetxt('{:}/keypoints_{:05d}.txt'.format(landmark_dir, i), keypoint[i].astype(np.uint8) / generator_output_size * side_lenth + [x_offset,y_offset] , fmt='%3i', delimiter=',')

print('Landmarks loaded.')

image_dir = 'deployment/vid2vid/datasets/face/test_img/0001/'
shutil.rmtree(image_dir)
os.mkdir(image_dir)
for file in os.listdir(source_image_dir):
	shutil.copyfile(os.path.join(source_image_dir,file), (os.path.join(image_dir,file)))

print('Images loaded.')

print('Running Vid2Vid.')
#os.system('deployment/run_vid2vid.sh')
print('Vid2Vid finished')

vid2vid_dir = 'deployment/vid2vid/results/edge2face_512/test_latest/0001'
for file in sorted(os.listdir(vid2vid_dir)):
	if 'fake_B' in file:
		full_path = os.path.join(vid2vid_dir, file)
		print(full_path)