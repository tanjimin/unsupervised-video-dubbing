import argparse
import cv2
import dlib
import torch
import librosa
import os
import subprocess
import numpy as np
import sys

from models import FaceEncoder, AudioEncoder, FaceDecoder

from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from main_support import step_1_main, step_2_main, step_4_main
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_step', type = int, help = 'Input 1 for running step1; 2 for step2; 3 for step3; 4 for step4 and etc.')
args = parser.parse_args()

int_step = args.run_step
shell_default = '/bin/zsh'

# ########################################
# Step 1: Generate keypoints
# ########################################

if int_step == 1:
    assert sys.executable.split('/')[-3] == 'capstone_vir'

    print('Step 1: Generating keypoints!')
    print('Please input base video file path !')
    base_video_file = input('===> ')
    assert os.path.isfile(base_video_file)

    print('Please input audio driver file path !')
    audio_driver_file = input('===> ')
    assert os.path.isfile(audio_driver_file)

    print('Please input epoch !')
    epoch = input('===> ')

    step_1_main(base_video_file, audio_driver_file, epoch)
    f = open("./result/source.txt","w+")
    f.write('Base Video: {}'.format(base_video_file + '\n\n'))
    f.write('Audio Driver: {}'.format(audio_driver_file+ '\n\n'))
    f.write('Epoch: {}'.format(epoch))
    f.close()
    print('Step 1 Done!')

# ########################################
# Step 2: Test generated images
# ########################################

if int_step == 2:
    print('Step 2: Test Image!')
    step2_load_keypoint = np.load('./result/keypoints_for_vis.npy')
    step_2_main(step2_load_keypoint)
    print('Step 2 Done!')

# ########################################
# Step 3: Exectute Vid2Vid
# ########################################

if int_step == 3:
    print('Step 3: Executing Vid2Vid')

    cmd_denoise = 'bash step_3_vid2vid.sh'
    shell = shell_default
    # subprocess.call([shell, '-c', cmd_denoise], stdout = open('/dev/null','w'), stderr = subprocess.STDOUT)
    subprocess.call([shell, '-c', cmd_denoise])

    if not os.path.exists('./result/vid2vid_frames'):
        os.mkdir('./result/vid2vid_frames')
    
    print('Please copy back generated images to /result/vid2vid_frames/')

# ########################################
# Step 4: Smooth generated images
# ########################################

if int_step == 4:
    print('Step 4: Smoothing vid2vid output')

    print("Please input Vid2Vid image folder path ! If default, enter 'd'!")
    vid2vid_path = input('===> ')

    if vid2vid_path == 'd':
        vid2vid_path = './result/vid2vid_frames/'

    print("Please input base image folder path ! If default, enter 'd'!")
    base_path = input('===> ')

    if base_path == 'd':
        base_path = './result/base_frames/'

    step_4_main(vid2vid_path, base_path, shell_default)


# ########################################
# Step 5: Concat modified images and audio
# ########################################

if int_step == 5:
    print('Step 5: Generate output!')

    print("Please input modified image folder path!")
    image_path = input('===> ')
    
    assert os.path.exists(image_path)
    assert image_path.endswith('/')

    print('Please input audio file path !')

    audio_driver_path = input('===> ')
    assert os.path.isfile(audio_driver_path)

    fps = 25.92002592002592
    
    step_5_main(shell_default, image_path, audio_driver_path, fps)

    print('Step 5 Done!')