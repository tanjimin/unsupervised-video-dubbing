from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *
import cv2
import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time
from PIL import ImageEnhance, ImageFilter
import random
from torchsummary import summary
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import torch.nn as nn
from IPython.display import Image
from models import * 
from utils.denoising_utils import *
from lrw_data_loader import LRW
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from copy import deepcopy
from torchvision import transforms
from PIL import Image
# left = 820
# top = 330
# right = 1030
# bottom = 540
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rect_frames', type = str, help = 'Input rect_frames path')
parser.add_argument('-c', '--circular_frames', type = str, help = 'Input circular_frames path')
args = parser.parse_args()

rect_frames_path = args.rect_frames
circular_frames_path = args.circular_frames

append_center_y = 310
append_center_x = 265
append_length = 70

#base_mouth_center_y = 430
#base_mouth_center_x = 930
#base_mouth_length = 100

base_mouth_center_y = 435
base_mouth_center_x = 925
base_mouth_length = 105

resize_ratio = base_mouth_length / float(append_length)

append_corr = [append_center_y - append_length, append_center_y + append_length, \
               append_center_x - append_length, append_center_x + append_length]

base_mouth_corr = [base_mouth_center_y - base_mouth_length, base_mouth_center_y + base_mouth_length, \
                   base_mouth_center_x - base_mouth_length, base_mouth_center_x + base_mouth_length]


def append_frame_to_base(video_base_path, video_append_frames_path, 
                         append_corr, base_mouth_corr, resize_ratio, folder_name):
    
    '''
    append_corr = [append_y1, append_y2, append_x1, append_x2]
    base_mouth_corr = [base_mouth_y1, base_mouth_y2, base_mouth_x1, base_mouth_x2]
    '''
    
    append_y1 = append_corr[0]
    append_y2 = append_corr[1]
    append_x1 = append_corr[2]
    append_x2 = append_corr[3]
    
    base_mouth_y1 = base_mouth_corr[0]
    base_mouth_y2 = base_mouth_corr[1]
    base_mouth_x1 = base_mouth_corr[2]
    base_mouth_x2 = base_mouth_corr[3]

    file_list = sorted(os.listdir(video_append_frames_path))
    file_list = [i for i in file_list if i != '.DS_Store']
    video_append_frames = []
    video_base = []
    video_base_list = os.listdir(video_base_path)
    video_base_list.sort()
    video_base_list = video_base_list[2:]
    
    for file in file_list:
#         print("File: {}".format(file[16:]))
#         frames = skvideo.io.vread(os.path.join(video_append_frames_path, file))
        frames = cv2.imread(os.path.join(video_append_frames_path, file))[...,::-1]
        video_append_frames.append(frames)

        video_to_append = np.array(video_append_frames)[:, append_y1:append_y2, append_x1:append_x2, :]
        # video_to_append: (84, 140, 140, 3)
        
        video_to_append_center_y = video_to_append_center_x = video_to_append_radius = 1/2 * video_to_append.shape[-2]
        mask = np.zeros(video_to_append.shape)
        threshold = int(video_to_append_radius)
        
        for x in range(int(video_to_append_radius) * 2):
            for y in range(int(video_to_append_radius) * 2):
                distance = (video_to_append_center_y - y) ** 2 +  (video_to_append_center_x - x) ** 2
                alpha = (1 - distance/(threshold ** 2)) * 3
                mask[:, x, y, :] = max(min(1, alpha), 0)

        video_base_frame = cv2.imread(os.path.join(video_base_path, 'frame'+file[17:]))[...,::-1]
        video_base.append(video_base_frame)
    
    video_base = np.array(video_base)
    
#     video_base = [to_tensor(Image.open(os.path.join(video_base_path, k))) for k in video_base_list if k != '.DS_Store']
#     video_base = torch.stack(video_base)
    # skvideo.io.vread(video_base_path)[2:-10, :]
    
#     print("V2A type: {}".format(type(video_to_append)))
#     print("VB type: {}".format(type(video_base)))
    assert video_to_append.shape[0] == video_base.shape[0]
    if not os.path.exists('./results'):
        os.mkdir('./results')
    
    dimension = (base_mouth_x2 - base_mouth_x1, base_mouth_y2 - base_mouth_y1)
    mask_resize = cv2.resize(mask[0], dsize=dimension, interpolation=cv2.INTER_LINEAR)
    target_mask = 1 - mask_resize

    for i in range(video_to_append.shape[0]):

        frame_to_append_resize = cv2.resize(video_to_append[i], \
                                            dsize=dimension, \
                                            interpolation=cv2.INTER_LINEAR) 
        
        video_base[i, base_mouth_y1:base_mouth_y2, base_mouth_x1:base_mouth_x2,:] = frame_to_append_resize * mask_resize + \
        video_base[i, base_mouth_y1:base_mouth_y2, base_mouth_x1:base_mouth_x2,:] * target_mask
        

        filename = 'frame{:05d}.png'.format(i)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        cv2.imwrite(os.path.join(folder_name, filename), video_base[i][...,::-1])
        
# noisy_dir = "/beegfs/yd1282/modified_frames/"
noisy_dir = rect_frames_path

# ground_orig_dir = "/beegfs/yd1282/original_frames/"
ground_orig_dir = "../result/base_frames/"

if not os.path.exists("../result/circular_smoothed_frames/"):
    append_frame_to_base(ground_orig_dir, "../result/vid2vid_frames/", 
                         append_corr, base_mouth_corr, 
                         resize_ratio, folder_name = "../result/circular_frames/")

# orig_dir = "/beegfs/yd1282/circular_smoothed_frames/"
orig_dir = circular_frames_path
noisy_files = os.listdir(noisy_dir)
noisy_files.sort()
orig_files = os.listdir(orig_dir)
orig_files.sort()
noisy_files = noisy_files[:74]

to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
# start = time.time()
# print("Starting at {}".format(start))
for n, o in zip(noisy_files, orig_files):
    
    print("Processing {}".format(o))
    noisy = os.path.join(noisy_dir, n)
    orig = os.path.join(orig_dir, o)

#     mouth_box = (left, top, right, bottom)

    # Add synthetic noise

    img_orig = get_image(orig, -1)[0]
    # img_pil = to_image(dn_data[0][1])
    width, height = img_orig.size
    # new_width, new_height = 256, 256
    left = 925 - (300 / 2)# (width - new_width) / 2
    top = 435 - (300 / 2)# (height - new_height) / 3
    right = 925 + (300 / 2)# (width + new_width) / 2
    bottom = 435 + (300 / 2)# (height + new_height) / 2.2 
    box = (int(left), int(top), int(right), int(bottom))

    base_mouth_center_y = 435
    base_mouth_center_x = 925
    base_mouth_length = 105

    img_pil = img_orig.crop((left, top, right, bottom))
    img_np = pil_to_np(img_pil)
#     print("IMG NP shape: {}".format(img_np.shape))

    img_noisy = get_image(noisy, -1)[0]
    # img_noisy_pil = to_image(dn_data[0][0])
    img_noisy_pil = img_noisy.crop((left, top, right, bottom))
    img_noisy_np = pil_to_np(img_noisy_pil)
#     print("IMG NOISY NP shape: {}".format(img_noisy_np.shape))
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 1500
    exp_weight=0.99

    num_iter = 1500
    input_depth = 3
    figsize = 5


    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=64, 
                  skip_n33u=64, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear', 
                  need_bias=False).type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    img_clean_torch = np_to_torch(img_np).type(dtype)
    net_input = img_noisy_torch # get_noise(input_depth, INPUT, 
                  #        (img_pil.size[1], 
                   #        img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    # s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    # print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0
    def closure():

        global i, out_avg, psrn_noisy_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    #     k = np.random.randint(0, len(dn_data))

    #     img_noisy_torch = to_tensor(to_image(dn_data[k][0]).crop((left, top, right, bottom))).type(dtype)
    #     print("Noisy dim: {}".format(img_noisy_torch.shape))
    #     img_np = to_tensor(to_image(dn_data[k][1]).crop((left, top, right, bottom))).type(dtype).cpu().numpy()
    #     img_noisy_np = img_noisy_torch.cpu().numpy()

        total_loss = mse(out, img_clean_torch)
        total_loss.backward()


        psrn_noisy = compare_psnr(img_noisy_np, 
                                  out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, 
                                  out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, 
                                  out_avg.detach().cpu().numpy()[0]) 

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        if i == (num_iter - 1):
            
            out_np = torch_to_np(out)
            out_img = to_image(torch.from_numpy(out_np).cpu())
#             print("Out img | type: {} | size: {}".format(type(out_img), out_img.size))
#             print("Box: {}".format(box))
            img_orig.paste(out_img, box)
            img_orig.save("../new_frames/" + o)
    #         plot_image_grid([np.clip(out_np, 0, 1), 
    #                          np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)



        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy

        i += 1

        return total_loss

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    
    print("Finished processing {}".format(o))

# end = time.time()
# print("Ending at {}".format(end))
# print("Time taken: {}".format(end - start))
