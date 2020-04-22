import cv2
import dlib
import torch

import os
import subprocess
#import skvideo.io
#import glob
import numpy as np
#import matplotlib.pyplot as plt
#from torchvision.utils import make_grid
#from PIL import Image

from charmodel import *

MAX_SEQ = 13
MAIN_PATH = '/scratch/yw3918/RNN_model/C6/C6_2.pt'
CHARENCODER_PATH = '/scratch/yw3918/RNN_model/C6/C6_2_charencoder.pt'

def frame_crop(input_frame):
    half_side_length = 350
    x_center = 960
    y_center = 400
    x_offset = x_center - half_side_length
    y_offset = y_center - half_side_length
    cropped_frame = input_frame[y_center - half_side_length: y_center + half_side_length,
                               x_center - half_side_length: x_center + half_side_length, :]
    rescaled_frame = cv2.resize(cropped_frame, (256, 256))
    return rescaled_frame

def detect_keypoints(frame):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
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
        print("Exception: Number of faces detected: {}".format(len(dets)))
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


# Keypoints extraction 
def extract_facial_keypoints(base_video):
    base_video_keypoints = []
    vidcap = cv2.VideoCapture(base_video)
    count = 0

    if not os.path.exists('./result/0001'):
        os.mkdir('./result/0001')
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if success:
            cv2.imwrite('./result/0001/sample_{:05d}.png'.format(count), frame)
            frame = frame_crop(frame)
            base_video_keypoints.append(detect_keypoints(frame))
            print('Frame {} processed'.format(count))
            count += 1
        else:
            break
    vidcap.release()
    print('Keypoints detection completed')
    base_video_facial_keypoints = np.array(base_video_keypoints)[:, :48, :]
    return base_video_facial_keypoints, count

def coordinate_to_matrix(coordinates):
    matrix = torch.zeros((29,256,256))
    for i,frame in enumerate(coordinates):
        for x,y in frame:
            matrix[i][x][y] = 1
    return matrix.float()

def pad_words(char_dict, max_seq, words):
    seq_tensors = [ [ char_dict.index(i) for i in word ] + [ len(char_dict) - 1 ] for word in words ]
    pre_pad_len = [ (max_seq - len(seq_tensor))//2 for seq_tensor in seq_tensors ]
    pst_pad_len = [ (max_seq - len(seq_tensor) + 1)//2 for seq_tensor in seq_tensors ]
    seq_tensors = [ [0] * pre_pad_len[i] + seq_tensor + [0] * pst_pad_len[i] for i,seq_tensor in enumerate(seq_tensors) ]
    return torch.LongTensor(seq_tensors) # batch * max_seq

def reload_model(model, path=""):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def keypoints_model(driver_label, base_video_facial_keypoints):
    char_dict = ['<pad>'] + [ _ for _ in string.ascii_uppercase ] + ['<eow>']
    max_seq = MAX_SEQ
    keypoints_matrix = coordinate_to_matrix(base_video_facial_keypoints[:29]).view( -1, 256, 256).unsqueeze(1) 
    print(keypoints_matrix.shape)
    seq_tensors = pad_words(char_dict, max_seq, [driver_label])
    print(seq_tensors.shape)
    model = CondRecurrentLG('eval', False)
    char_encoder = CharacterEncoder()
    model = reload_model(model, MAIN_PATH).eval()
    char_encoder = reload_model(char_encoder, CHARENCODER_PATH).eval()

    seq_tensors = char_encoder(seq_tensors)
    target_mouth_keypoints, _, __ = model(keypoints_matrix, seq_tensors)
    target_mouth_keypoints = target_mouth_keypoints.view(-1, 20, 2).squeeze(0)
    return target_mouth_keypoints

def combine_keypoints(base_video_facial_keypoints, target_mouth_keypoints):
    facial_kp_tensor = torch.FloatTensor(base_video_facial_keypoints[:29])
    generated_keypoints = torch.cat((facial_kp_tensor, target_mouth_keypoints), dim = 1)
    return generated_keypoints

def generate_keypoints(base_video, driver_label):
    base_video_facial_keypoints, num_of_frames = extract_facial_keypoints(base_video)
    target_mouth_keypoints = keypoints_model(driver_label, base_video_facial_keypoints)
    generated_keypoints = combine_keypoints(base_video_facial_keypoints, target_mouth_keypoints) 
    return generated_keypoints


def step_1_main(base_video_file, driver_label):
    dlib.DLIB_USE_CUDA = True

    # keypoints = generate_keypoints('./source/base_video/spanish_3sec.mp4', \
    #                                './source/audio_driver_wav/english_3sec_7.wav')
    keypoints = generate_keypoints(base_video_file, driver_label)

    if not os.path.exists('./result'):
        os.mkdir('./result')
    np.save('./result/keypoints', keypoints.cpu().detach().numpy())

if __name__ == "__main__":
    filename = 'familias.avi'
    driver_label = 'family'.upper()
    step_1_main(filename, driver_label)
    
