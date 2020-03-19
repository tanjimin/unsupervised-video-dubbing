from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.utils.data
from numpy import genfromtxt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

import cv2
from data import *
from charmodel import *
from torchvision import datasets, transforms
from torchvision.utils import save_image

SEED = 1
FORMAT = '%(asctime)-15s %(message)s'

class Arguments():
    def __init__(self):
        self.batch_size = 2
        self.epochs = 50
        self.lr = 0.001
        self.b1 = 0.5
        self.b2 = 0.999
        self.vocab_size = 1
        self.interval = 50
        self.seed = 1
        self.log_interval = 50
        self.no_cuda = False
        self.mode = 'C3'
        self.path = '/scratch/yw3918/RNN_model/C3/C3_4.pt'
        self.data_dir = '/scratch/yw3918/RNN_both/'
        self.workers = 1
        self.test = True


def transformer(classes, id):
    char_dict = ['<pad>'] + [ _ for _ in string.ascii_uppercase ] + ['<eow>']
    max_seq = 13
    word = classes[id]
    seq_tensor  = [ char_dict.index(i) for i in word ] + [ len(char_dict) - 1 ] 
    pre_pad_len = (max_seq - len(seq_tensor))//2 
    pst_pad_len = (max_seq - len(seq_tensor) + 1)//2 
    seq_tensor  = [0] * pre_pad_len + seq_tensor + [0] * pst_pad_len 
    seq_tensor = torch.tensor(seq_tensor)
    return seq_tensor


def data_loader(args):
    # dset_folders = {x: datasets.DatasetFolder(os.path.join(args.data_dir, x), 
    #     np.load, extensions='npz',target_transform = lambda id: transformer(self.classes,id)) for x in ['train', 'val', 'test']}
    # for x, datafolder in dset_folders.items():
    #     datafolder.target_transform = lambda id: transformer(datafolder.classes,id)
    dsets = {x: LRW(x) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,\
                       shuffle=True, num_workers=args.workers) for x in ['train', 'val', 'test']}
    # dset_loaders = {x: torch.utils.data.DataLoader( dset_folders[x],
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.workers) for x in ['train', 'val', 'test']}
    return dset_loaders

def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model

class AdjustLR(object):
    def __init__(self, optimizer, init_lr, sleep_epochs=5, half=5, verbose=0):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr
        self.verbose = verbose

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr
            if self.verbose:
                print('>>> reduce learning rate <<<')


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

# def criterion(output_x, x, mu, logsig):
#     reshape_x = x.view(-1, 29 * 40)
#     recon_f = nn.MSELoss(reduction = 'mean')
#     recon_loss = recon_f(output_x.view(-1, 29 * 40), reshape_x) 
#     KLD_element = logsig - mu**2 - torch.exp(logsig) + 1 
#     KLD = - torch.mean(torch.sum(KLD_element* 0.5, dim=2) )
#     #0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
#     loss = recon_loss #+ KLD
#     return loss


def criterion(output_x, x, mu, logsig):
    reshape_x = x.view(-1, 29 * 40)
    output_x = output_x.view(-1,29,20,2)

    recon_f = nn.MSELoss(reduction = 'mean')
    recon_loss = recon_f(output_x.view(-1, 29 * 40), reshape_x) 

    KLD_element = logsig - mu**2 - torch.exp(logsig) + 1 
    KLD = - torch.mean(torch.sum(KLD_element* 0.5, dim=2) )
    #0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)

    loss = recon_loss #+ KLD
    return loss

def train_test(char_encoder, model, dset_loaders, epoch, phase, optimizer, args, logger, use_gpu, save_path):
    if phase == 'val' or phase == 'test':
        model.eval()
    if phase == 'train':
        model.train()
    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    char_dict = ['<pad>'] + [ _ for _ in string.ascii_uppercase ] + ['<eow>']
    max_seq = 13
    running_loss, running_corrects, running_all = 0., 0., 0.
    results = []
    for batch_idx, (train_data, seq_tensors, labels) in enumerate(dset_loaders[phase]):
        train_inputs, train_targets = train_data['train_inputs'], train_data['train_targets']
        train_inputs, train_targets = train_inputs.float(), train_targets.float()
        # 2 * 29 * 256 * 256
        train_inputs = train_inputs.view( -1, 256, 256).unsqueeze(1) 
        # pad labels/words
        #seq_tensors = pad_words(char_dict, max_seq, label)

        if use_gpu:
            train_inputs = train_inputs.cuda()
            train_targets = train_targets.cuda()
            seq_tensors = seq_tensors.cuda()
            #label = label.cuda()
            char_encoder = char_encoder.cuda()
            model = model.cuda()


        #train_targets = train_targets.view(-1, 29, 40) 
        if phase == 'train':
            seq_tensors = char_encoder(seq_tensors)
            train_outputs, mu, sig = model(train_inputs, seq_tensors) # output: 2*29 * 40
            loss = criterion(train_outputs, train_targets, mu, sig)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                seq_tensors = char_encoder(seq_tensors)
                train_outputs, mu, sig = model(train_inputs, seq_tensors)
                loss = criterion(train_outputs, train_targets, mu, sig)
                optimizer.zero_grad()
                results.append([train_inputs.detach().cpu().numpy(), train_outputs.detach().cpu().numpy(), labels])
        # stastics
        running_loss += loss.data * train_inputs.size(0)
        running_all += len(train_inputs)
        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_loss / running_all,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        phase,
        epoch,
        running_loss / len(dset_loaders[phase].dataset),
        running_corrects / len(dset_loaders[phase].dataset))+'\n')
    if phase == 'train':
        torch.save(model.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.pt')
        return model
    if phase == 'test':
        return results


def test_adam(args, use_gpu):
    save_path = '/scratch/yw3918/RNN_model/' +  args.mode
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+args.mode+'_'+str(args.lr)+'.txt'
    logger_name = "mylog"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    char_encoder = CharacterEncoder()
    model = CondRecurrentLG(mode=args.mode, use_gpu =use_gpu )
    # reload model
    model = reload_model(model, logger, args.path)
    # define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)

    dset_loaders = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
    if args.test:
        # train_test(model, dset_loaders, 0, 'val', optimizer, args, logger, use_gpu, save_path)
        results = train_test(char_encoder, model, dset_loaders, 0, 'test', optimizer, args, logger, use_gpu, save_path)
        return results
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        model = train_test(char_encoder, model, dset_loaders, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(char_encoder, model, dset_loaders, epoch, 'val', optimizer, args, logger, use_gpu, save_path)


if __name__ == '__main__':
    args = Arguments()
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    use_gpu = torch.cuda.is_available()
    results = test_adam(args, use_gpu)
    np.save(open('results.npy','wb'), results)
