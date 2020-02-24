import os
import glob
import numpy as np
import torch

class LRW():

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
            
        self.video_file = os.path.join(self.root_path, 'video_' + self.folds+ '.npy')
        self.audio_file = os.path.join(self.root_path, 'audio_' + self.folds +'.npy')
  
        self.video = npy_loader(self.video_file)
        self.audio = npy_loader(self.audio_file)
        
        print('Loading {} part'.format(self.folds))

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        vid = self.augmentation(self.video[idx, :, :, :])
        aud = self.audio[idx, :, :]
        labels = 0
        return (vid, aud), labels    
        
    def augmentation(self, keypoints):
        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        if self.folds == 'train':
            d = keypoints_move + ones * randint
        else:
            d = keypoints_move + ones * 38
        return d

def npy_loader(file):
    return torch.tensor(np.load(file)).float()
