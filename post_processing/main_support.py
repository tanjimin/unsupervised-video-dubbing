import cv2
import dlib
import torch
import librosa
import os
import subprocess
import skvideo.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image

from models import FaceEncoder, AudioEncoder, FaceDecoder

# #############################################################
# ## Step 1
# #############################################################
def step_1_main(base_video_file, audio_driver_file, epoch):
    dlib.DLIB_USE_CUDA = True
    if not os.path.exists('./result'):
        os.mkdir('./result')

    # keypoints = generate_keypoints('./source/base_video/spanish_3sec.mp4', \
    #                                './source/audio_driver_wav/english_3sec_7.wav')
    keypoints, scale_offset = generate_keypoints(base_video_file, audio_driver_file, epoch)
    np.save('./result/keypoints_for_vis', keypoints.cpu().detach().numpy())

    assert keypoints.shape[0] == len(scale_offset.keys())


    # scale_offset = {'0':([x_offset, y_offset],scale_ratio), ...}
    for idx, elem in enumerate(keypoints):
        keypoints[idx] = elem * scale_offset[idx][1] + torch.Tensor(scale_offset[idx][0])

    np.save('./result/keypoints', keypoints.cpu().detach().numpy())

    # extract frames from base video
    # extract_frames(base_video_file)
    #keypoints_model(np.zeros((29, 12)), np.zeros((29, 48, 2)))
    #print(a.shape)
    pass


def generate_video(base_video, driver_video):
    generated_keypoints = generate_keypoints(base_video, driver_video, epoch)
    generated_frames = vid2vid(base_video, generate_keypoints) # First frame of base_video actually
    replaced_frames = replace_mouth(base_video, generated_frames)
    final_video = align_audio(replaced_frames, base_video)
    return final_video

######################################################
## Generate Keypoints from base video and driver audio
######################################################

def generate_keypoints(base_video, driver_video, epoch):
    base_video_facial_keypoints, num_of_frames, scale_offset = extract_facial_keypoints(base_video)
    driver_audio_feature = extract_mfcc(driver_video, num_of_frames)
    target_mouth_keypoints = keypoints_model(driver_audio_feature, base_video_facial_keypoints, epoch)
    generated_keypoints = combine_keypoints(base_video_facial_keypoints, target_mouth_keypoints) 
    return generated_keypoints, scale_offset

# mfcc extraction
def extract_mfcc(driver_video, num_of_frames):
    audio, sr = librosa.load(driver_video, sr = 16000)
    audio_samples = audio.shape[0]
    specified_hop_length = int(audio_samples / num_of_frames) + 1
    mfccs = librosa.feature.mfcc(audio, sr = sr, n_mfcc = 13, win_length = 3 * specified_hop_length, hop_length = specified_hop_length, center = True)
    mfccs_features = mfccs[1:] # First feature is offset, thus removed
    driver_audio_feature = mfccs_features.transpose((1, 0))
    print('Driver audio feature completed with length: ', driver_audio_feature.shape[0])
    return driver_audio_feature

# Keypoints extraction 
def extract_facial_keypoints(base_video):
    if not os.path.exists('./result/base_frames'):
        os.mkdir('./result/base_frames')
    base_video_keypoints = []
    saved_offset_scale = {}
    vidcap = cv2.VideoCapture(base_video)
    count = 0
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if success:
            filename = 'frame_{:05d}.png'.format(count)
            cv2.imwrite('./result/base_frames/frame_{:05d}.png'.format(count), frame)
            keypoints = detect_keypoints(frame, filename)

            ######################################################
            # remember to minus 1 for indexing

            # up = min(20,25) --- y [1]
            # down = 9 --- y [1]
            # left = min(1,2,3) --- x [0]
            # right = max(15,16,17) --- x[0]
            offset_ratio = 0.25

            up = min(keypoints[20-1][1], keypoints[25-1][1])
            down = keypoints[9][1]
            left = min(keypoints[1-1][0], keypoints[2-1][0], keypoints[3-1][0])
            right = max(keypoints[15-1][0], keypoints[16-1][0], keypoints[17-1][0])

            up_offset = int(up - up * offset_ratio)
            down_offset = int(down + down * offset_ratio)
            left_offset = int(left - left * offset_ratio)
            right_offset = int(right + right * offset_ratio)

            center = [*map(int,(0.5 * (right_offset - left_offset) + left_offset, 0.5 * (down_offset - up_offset) + up_offset))]
            crop_length = int(max(down_offset - up_offset, right_offset-left_offset) / 2)
            
            up_crop = y_offset = center[1] - crop_length
            down_crop = center[1] + crop_length
            left_crop = x_offset = center[0] - crop_length
            right_crop = center[0] + crop_length

            scale_ratio = crop_length * 2 / 256
            frame_to_model = (keypoints - np.array([x_offset, y_offset])) / scale_ratio

            # TODO
            # Save x_offset, y_offset, scale_ratio
            saved_offset_scale[count] = ([x_offset, y_offset],scale_ratio)
            ######################################################
            base_video_keypoints.append(frame_to_model)

            print('Frame {} processed'.format(count))
            count += 1
        else:
            break
    vidcap.release()
    print('Keypoints detection completed')
    base_video_facial_keypoints = np.array(base_video_keypoints)[:, :48, :]
    return base_video_facial_keypoints, count, saved_offset_scale

# def frame_crop(input_frame):
#     half_side_length = 350
#     x_center = 960
#     y_center = 400
#     x_offset = x_center - half_side_length
#     y_offset = y_center - half_side_length
#     cropped_frame = input_frame[y_center - half_side_length: y_center + half_side_length,
#                                x_center - half_side_length: x_center + half_side_length, :]
#     rescaled_frame = cv2.resize(cropped_frame, (256, 256))
#     return rescaled_frame


# Dlib for keypoints detection
# def detect_keypoints(frame):
#     predictor_path = './source/dlib/shape_predictor_68_face_landmarks.dat'
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(predictor_path)
#     dets = detector(frame, 1)
#     keypoints = []
#     if len(dets) == 1:
#         for k, d in enumerate(dets):
#             shape = predictor(frame, d)
#             face = range(0, 68)
#             for i in face:
#                 x = shape.part(i).x
#                 y = shape.part(i).y
#                 keypoints.append([x, y])
#         return np.array(keypoints)
#     else:
#         raise Exception("Number of faces detected for {}: {}".format(filename, len(dets)))
def detect_keypoints(frame, filename):
    predictor_path = './source/dlib/shape_predictor_68_face_landmarks.dat'
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
        print("Input an integer to choose one face (START from 0): {}".format(dets))
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

    
# Encoder-Decoder to generate mouth keypoints
def keypoints_model(driver_audio_feature, base_video_facial_keypoints, epoch):
    
    face_encoder = torch.load('./source/model/face_encoder_{}.pt'.format(str(epoch)), map_location=torch.device('cpu')).eval()
    audio_encoder = torch.load('./source/model/audio_encoder_{}.pt'.format(str(epoch)), map_location=torch.device('cpu')).eval()
    face_decoder = torch.load('./source/model/face_decoder_{}.pt'.format(str(epoch)), map_location=torch.device('cpu')).eval()
    face_embedding = face_encoder(torch.FloatTensor(base_video_facial_keypoints).reshape(-1, 96))
    audio_embedding = audio_encoder(torch.FloatTensor(driver_audio_feature))
    embedding = torch.cat((face_embedding, audio_embedding), dim = 1).view(1, -1, 144)
    target_mouth_keypoints = (face_decoder(embedding) * 255).reshape(-1, 20, 2).squeeze(0)
    # [86, 20, 2]
    # base_video_facial_keypoints: (86, 48, 2)
    return target_mouth_keypoints

def combine_keypoints(base_video_facial_keypoints, target_mouth_keypoints):
    facial_kp_tensor = torch.FloatTensor(base_video_facial_keypoints)
    generated_keypoints = torch.cat((facial_kp_tensor, target_mouth_keypoints), dim = 1)
    return generated_keypoints


# #############################################################
# ## Step 2 Test Image
# #############################################################
def image_from_tensor(tensor):
    batch_size = tensor.shape[0]
    img = np.zeros((batch_size,256,256,3), np.uint8) + 128
    for i in range(batch_size):
        for row in range(67):
            x_1 = tensor[i, row, 0]
            y_1 = tensor[i, row, 1]
            x_2 = tensor[i, row+1, 0]
            y_2 = tensor[i, row+1, 1]
            cv2.circle(img[i], (x_1, y_1), 1, (0, 0, 255), -1)
            cv2.circle(img[i], (x_2, y_2), 1, (0, 0, 255), -1)
            cv2.line(img[i], (x_1, y_1), (x_2, y_2), (0, 0, 255), 1)
    img = np.transpose(img, (0, 3, 1, 2))
    return torch.tensor(img)

def step_2_main(keypoint):
    
    keypoint_image = image_from_tensor(keypoint)

    key_image_save = './result/keypoints_frames'
    if not os.path.exists(key_image_save):
        os.mkdir(key_image_save)

    save_keypoints_csv = './result/save_keypoints_csv'
    if not os.path.exists(save_keypoints_csv):
        os.mkdir(save_keypoints_csv)

    generator_output_size = 256
    half_side_length = 350
    side_lenth = half_side_length * 2
    x_center = 960
    y_center = 400

    x_offset = x_center - half_side_length
    y_offset = y_center - half_side_length


    def save_image_(tensor, fp, nrow=8, padding=2,
                   normalize=False, range=None, scale_each=False, pad_value=0, format=None):
        """Save a given Tensor into an image file.

        Args:
            tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
                saves the tensor as a grid of images by calling ``make_grid``.
            fp - A filename(string) or file object
            format(Optional):  If omitted, the format to use is determined from the filename extension.
                If a file object was used instead of a filename, this parameter should always be used.
            **kwargs: Other arguments are documented in ``make_grid``.
        """

        grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(fp, format=format)

    for i in range(86):
        save_image_(keypoint_image[i],os.path.join(key_image_save, 'vis_{:05d}.png'.format(i)))
        np.savetxt(os.path.join(save_keypoints_csv, 'keypoints_{:05d}.txt'.format(i)), keypoint[i].astype(np.uint8) / generator_output_size * side_lenth + [x_offset,y_offset] , fmt='%3i', delimiter=',')


    upper_lip = keypoint[:, 66]
    u_x = upper_lip[:, 0]
    u_y = upper_lip[:, 1]

    lower_lip = keypoint[:, 62]
    l_x = lower_lip[:, 0]
    l_y = lower_lip[:, 1]

    openness = np.sqrt((u_x - l_x) ** 2 + (u_y - l_y) ** 2)
    
    plt.plot(openness)
    plt.ylim(0, 15)
    plt.savefig('./result/openness.png')

# #############################################################
# ## Step 3 vid2vid
# #############################################################

# step_3_vid2vid.sh

# #############################################################
# ## Step 4 smooth output
# #############################################################

def auto_copy_paste(vid_list, base_list):

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
        
        if not os.path.exists('./result/modified_frames'):
            os.mkdir('./result/modified_frames')
            
        cv2.imwrite(os.path.join('./result/modified_frames', filename), base_frame[...,::-1])
        
        print(filename, 'processed!')
        print('Modified images are saved at ./result/modified_frames/')

def step_4_main(fake_image_path, orig_image_path, shell_default):
    vid_list = sorted(glob.glob(os.path.join(fake_image_path, '*.jpg')))
    base_list = sorted(glob.glob(os.path.join(orig_image_path, '*.jpg')))
    assert len(vid_list) == len(base_list)
    
    auto_copy_paste(vid_list, base_list)
    
    cmd_denoise = 'bash step_4_denoise.sh'
    shell = shell_default
    subprocess.call([shell, '-c', cmd_denoise], stdout = open('/dev/null','w'), stderr = subprocess.STDOUT)


# #############################################################
# ## Step 5 generate output
# #############################################################
def step_5_main(shell_default, image_path, audio_driver_path, fps):
    if not os.path.exists(image_path):
        print('Image path does not exists!')
        return

    if not os.path.isfile(audio_driver_path):
        print('Audio file does not exists!')
        return


    if not os.path.exists(os.path.join(image_path, 'result')):
        os.mkdir(os.path.join(image_path, 'result'))
    
    make_video_from_frames(image_path, fps, shell_default)
    print('Video without audio is created!')
    extract_audio(image_path, audio_driver_path, shell_default)
    print('Audio is extracted!')
    combine_audio_video(image_path, shell_default)


def make_video_from_frames(image_path, fps, shell_default):
    
    # file_list = sorted(os.listdir(image_path))
    # video_frame = []
    # for file in sorted(glob.glob(os.path.join(image_path, "*.png"))):
        
    #     videodata = skvideo.io.vread(file)
    #     video_frame.append(videodata)
    
    # videos = np.array(video_frame)[:,0]
    
    
    # skvideo.io.vwrite(os.path.join(image_path, './result/video_without_audio.mp4'), videos)

    # ffmpeg -framerate 25.92002592002592 -i frame%05d.png -pix_fmt yuv420p video_without_audio.mp4
    image = image_path + 'vis_%05d.png'
    save_loc = os.path.join(image_path, 'result/video_without_audio.mp4')
    cmd_make_video = 'ffmpeg -framerate ' + str(fps) + ' -i ' + image + ' -pix_fmt yuv420p ' + save_loc

    shell = shell_default
    subprocess.call([shell, '-c', cmd_make_video], stdout = open('/dev/null','w'), stderr = subprocess.STDOUT)


# extract audio from the audio driver
def extract_audio(image_path, audio_driver_path, shell_default):

    audio_output = os.path.join(image_path, './result/audio_only.mp4')
    
    # cmd_extract_audio = 'ffmpeg -i ' + audio_driver_path + ' -ab 160k -ac 2 -ar 16000 -vn ' + audio_output
    cmd_extract_audio = 'ffmpeg -i ' + audio_driver_path + ' -ac 2 -ar 48000 -vn ' + audio_output
    shell = shell_default

    subprocess.call([shell, '-c', cmd_extract_audio], stdout = open('/dev/null','w'), stderr = subprocess.STDOUT)
    
# combine the two streams together (new audio with originally exisiting video)
def combine_audio_video(image_path, shell_default):

    video_input = os.path.join(image_path, './result/video_without_audio.mp4')
    audio_input = os.path.join(image_path, './result/audio_only.mp4')
    output = os.path.join(image_path, './result/final_output.mp4')

    # cmd_combine = 'ffmpeg -i ' + video_input + ' -i ' + audio_input + ' -shortest -c:v copy -c:a aac -b:a 256k ' + output
    cmd_combine = 'ffmpeg -i ' + video_input + ' -i ' + audio_input + ' -c:v copy -c:a aac -b:a 256k ' + output
    shell = shell_default
    subprocess.call([shell, '-c', cmd_combine], stdout = open('/dev/null','w'), stderr = subprocess.STDOUT)



    


# #############################################################
# ## Style transfer from keypoints to video frames with vid2vid
# #############################################################

# def vid2vid(base_video, generate_keypoints):

#     generated_frames = None

#     return generated_frames


# def replace_mouth(base_video, generated_frames):

#     replaced_frames = None

#     return replaced_frames

# def align_audio(replaced_frames, base_video):

#     final_video = None

#     return final_video
