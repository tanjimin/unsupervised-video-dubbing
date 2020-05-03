import cv2
import dlib
import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
import os


vid_list = sorted(glob.glob(os.path.join('./vid2vid_frames', '*.jpg')))
base_list = sorted(glob.glob(os.path.join('./base_frames', '*.jpg')))
assert len(vid_list) == len(base_list)


def detect_keypoints(frame, filename):
    predictor_path = '../source/dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(frame, 1)
    keypoints = []
    if len(dets) == 1:
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            face = range(0, 68)
            for i in face:
                x = shape.part(i).x
                y = shape.part(i).y
                keypoints.append([x, y])
        return np.array(keypoints)
    else:
        print('-'*30)
        print("Exception: Number of faces detected for {}: {}".format(filename, len(dets)))
        print("Input an integer to choose one face: {}".format(dets))
        chosen_face = int(input())
        print('-'*30)
        for k, d in enumerate(dets):
            if k == chosen_face:
                shape = predictor(frame, d)
                face = range(0, 68)
                for i in face:
                    x = shape.part(i).x
                    y = shape.part(i).y
                    keypoints.append([x, y])
        return np.array(keypoints)


for vid_img, base_img in zip(vid_list, base_list):
    # ===========================================================================
    # Vid Image
    # ===========================================================================
    # (512, 512, 3)
    vid_frames = np.array(cv2.imread(vid_img)[...,::-1])
    filename = vid_img.split('/')[-1]
    
    # (68, 2)
    vid_keys = detect_keypoints(vid_frames, filename)
    
    # for each frame, get mouth(up,down,left,right) coordinates
    left_most = 49 - 1
    right_most = 55 - 1 
    # bottom = max(49, 55 ~ 60) --- 48, range(54, 59+1)
    # top = min (49 ~ 55) --- range(48, 54+1)
    
    # face[left, right, bottom, top]
    face_rect = [6 - 1, 12 - 1, 9 - 1, 3 - 1]
    # get face rect region
    vid_face_rect_l = vid_keys[face_rect[0]][0]
    vid_face_rect_r = vid_keys[face_rect[1]][0]
    vid_face_rect_b = vid_keys[face_rect[2]][1]
    vid_face_rect_t = vid_keys[face_rect[3]][1]
    vid_face_rect_width = vid_face_rect_r - vid_face_rect_l
    vid_face_rect_height = vid_face_rect_b - vid_face_rect_t
    
    vid_left_most_coor = (vid_keys[left_most][0] - vid_face_rect_l) / vid_face_rect_width
    vid_right_most_coor = (vid_keys[right_most][0] - vid_face_rect_l) / vid_face_rect_width
    bottom_corr_1 = [vid_keys[i][1] for i in range(54,60)]
    bottom_corr_1.append(vid_keys[48][1])
    vid_bottom_corr = (max(bottom_corr_1) - vid_face_rect_t) / vid_face_rect_height
    vid_top_corr = (min([vid_keys[i][1] for i in range(48,55)]) - vid_face_rect_t) / vid_face_rect_height
    
    # ===========================================================================
    # Base Frame
    # ===========================================================================
    base_frame = np.array(cv2.imread(base_img)[...,::-1])
    filename = base_img.split('/')[-1]
    
    base_keys = detect_keypoints(base_frame, filename)
    
    # get face rect region
    base_face_rect_l = base_keys[face_rect[0]][0]
    base_face_rect_r = base_keys[face_rect[1]][0]
    base_face_rect_b = base_keys[face_rect[2]][1]
    base_face_rect_t = base_keys[face_rect[3]][1]
    base_face_rect_width = base_face_rect_r - base_face_rect_l
    base_face_rect_height = base_face_rect_b - base_face_rect_t
    
    base_left_most_coor = (base_keys[left_most][0] - base_face_rect_l) / base_face_rect_width
    base_right_most_coor = (base_keys[right_most][0] - base_face_rect_l) / base_face_rect_width
    bottom_corr_11 = [base_keys[i][1] for i in range(54,60)]
    bottom_corr_11.append(base_keys[48][1])
    base_bottom_corr = (max(bottom_corr_11) - base_face_rect_t) / base_face_rect_height
    base_top_corr = (min([base_keys[i][1] for i in range(48,55)])  - base_face_rect_t)/ base_face_rect_height
    
    # ===========================================================================
    # Compare normalized coordinates
    # ===========================================================================
    buffer = 0.1
    left_most_corr = min(vid_left_most_coor, base_left_most_coor) * (1 - buffer)
    right_most_corr = max(vid_right_most_coor, base_right_most_coor) * (1 + buffer)
    bottom_most_corr = max(vid_bottom_corr, base_bottom_corr) * (1 + buffer)
    top_most_corr = min(vid_top_corr, base_top_corr) * (1 - buffer)
    
    # vid regions
    vid_l = int(left_most_corr * vid_face_rect_width) + vid_face_rect_l
    vid_r = int(right_most_corr * vid_face_rect_width) + vid_face_rect_l
    vid_b = int(bottom_most_corr * vid_face_rect_height) + vid_face_rect_t
    vid_t = int(top_most_corr * vid_face_rect_height) + vid_face_rect_t
    vid_cropped = vid_frames[vid_t:vid_b, vid_l:vid_r]
    
    base_l = int(left_most_corr * base_face_rect_width) + base_face_rect_l
    base_r = int(right_most_corr * base_face_rect_width) + base_face_rect_l
    base_b = int(bottom_most_corr * base_face_rect_height) + base_face_rect_t
    base_t = int(top_most_corr * base_face_rect_height) + base_face_rect_t
    
    # calculate dim for resizing
    base_y = base_b - base_t
    base_x = base_r - base_l

    # get the corresponding cropped frames
    frame_to_append_resize = cv2.resize(vid_cropped, dsize=(base_x, base_y), interpolation=cv2.INTER_LINEAR) 
    
    base_frame[base_t:base_b, base_l:base_r] = frame_to_append_resize
#     plt.imshow(base_frame)
#     plt.show()
    
    if not os.path.exists('./modified_frames'):
        os.mkdir('./modified_frames')
        
    cv2.imwrite(os.path.join('./modified_frames', filename), base_frame[...,::-1])
    
    print(filename, 'processed!')