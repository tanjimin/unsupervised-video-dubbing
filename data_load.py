import os
import glob
import numpy as np
import torch

class LRW_Dataset_AV():

    def __init__(self, folds,
                 labels_file = './data/label_sorted.txt', 
                 root_path = '/beegfs/cy1355/lipread_datachunk_big/', 
                 transform = None):
        """
        Args:
            labels_file (string): Path to the text file with labels
            root_path (string): Path to the file with the facial landmarks and audio features (MFCC)
            folds (string): train / val / test indicator
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        self.folds = folds
        self.labels_file = labels_file
        self.root_path = root_path
        with open(self.labels_file) as myfile:
            self.data_dir = myfile.read().splitlines()
            
        self.v_list = {}
        self.a_list = {}
        
        self.video_file = os.path.join(self.root_path, 'video_' + self.folds+ '.npy')
        self.audio_file = os.path.join(self.root_path, 'audio_' + self.folds +'.npy')
  
        if self.folds == 'test':
            self.video = npy_loader_aug_test(self.video_file, v_flag = 1)
            self.audio = npy_loader_aug_test(self.audio_file, v_flag = 0)
        else:
            self.video = npy_loader_aug(self.video_file, v_flag = 1)
            self.audio = npy_loader_aug(self.audio_file, v_flag = 0)
            
        self.labels = 0
        
        print('Loading {} part'.format(self.folds))

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
            
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        vid = self.video[idx, :, :, :]
        aud = self.audio[idx, :, :]
        labels = 0
        
        sample = (vid, aud), labels
        
        return sample

def npy_loader_aug(file, v_flag):
    
    data = np.load(file)
    if v_flag == 1:
        keypoints = torch.tensor(data).float()

        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        d = keypoints_move + ones * randint
    
    else:
        d = torch.tensor(data).float()
    return d

def npy_loader_aug_test(file, v_flag):
    
    data = np.load(file)
    if v_flag == 1:
        keypoints = torch.tensor(data).float()

        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        d = keypoints_move + ones * 38
    
    else:
        d = torch.tensor(data).float()
    return d
