
import matplotlib.pyplot as plt
import os
import numpy as np
import random

import cv2
import torch
import torch.optim
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torchvision import transforms

from PIL import Image
from skimage.measure import compare_psnr
from denoising_utils.denoising_utils import *
from denoising_utils.common_utils import *
from denoising_models import *

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def DIP_denoise(rect_frame, circular_frame, out_frame, base_center_x, base_center_y, num_iter = 2):

    '''
    Arguments:

    rect_frame: full path to a single rectangular paste image
    circular_frame: full path to the corresponding circular smoothed image
    orig_frame: full path to the original reference frame (from source video)
    out_frame: full path to final smoothed frame 
    num_iter: number of passes through DIP network
    '''
    left = base_center_x - (300 / 2)
    top = base_center_y - (300 / 2)
    right = base_center_x + (300 / 2)
    bottom = base_center_y + (300 / 2)
    box = (int(left), int(top), int(right), int(bottom))
    
    print("Processing {}".format(out_frame))
    
    # Fetch images from paths. The circular frame is the reference frame for smoothing.

    img_orig = get_image(circular_frame, -1)[0]
    img_pil = img_orig.crop((left, top, right, bottom))
    img_np = pil_to_np(img_pil)

    img_noisy = get_image(rect_frame, -1)[0]
    img_noisy_pil = img_noisy.crop((left, top, right, bottom))
    img_noisy_np = pil_to_np(img_noisy_pil)

    # Initialize parameters for deep image prior network

    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./30. 
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = num_iter
    exp_weight=0.99

    num_iter = num_iter
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
    net_input = img_noisy_torch 

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0

    def closure():

        nonlocal i, out_avg, psrn_noisy_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_clean_torch)
        total_loss.backward()


        psrn_noisy = compare_psnr(img_noisy_np, 
                                out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, 
                                out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, 
                                out_avg.detach().cpu().numpy()[0]) 

        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        if i == (num_iter - 1):
            
            out_np = torch_to_np(out)
            out_img = to_image(torch.from_numpy(out_np).cpu())

            img_orig.paste(out_img, box)
            img_orig.save(out_frame)
    
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

    print("Finished processing {}".format(out_frame))