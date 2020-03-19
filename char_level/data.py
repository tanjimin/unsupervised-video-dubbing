import os
import glob
import string
import numpy as np
import torch

def pad_words(char_dict, max_seq, words):
    seq_tensors = [ [ char_dict.index(i) for i in word ] + [ len(char_dict) - 1 ] for word in words ]
    pre_pad_len = [ (max_seq - len(seq_tensor))//2 for seq_tensor in seq_tensors ]
    pst_pad_len = [ (max_seq - len(seq_tensor) + 1)//2 for seq_tensor in seq_tensors ]
    seq_tensors = [ [0] * pre_pad_len[i] + seq_tensor + [0] * pst_pad_len[i] for i,seq_tensor in enumerate(seq_tensors) ]
    return torch.LongTensor(seq_tensors) # batch * max_seq

def matrix_coordinate_transform(coordinates):
    matrix = torch.zeros((29,256,256))
    for i,frame in enumerate(coordinates):
        for x,y in frame:
            matrix[i][x][y] = 1
    return matrix 


class LRW():

    def __init__(self, folds,
                 labels_file = '/home/yw3918/Capstone/Char_model/labels_sum.npy', 
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
        
        self.labels = np.load(open(labels_file,'rb'))
        self.labels_name = self.labels[:,0]
        self.labels_interval = self.labels[:,1].astype(np.int)
            
        self.video_file = os.path.join(self.root_path, 'video_' + self.folds+ '.npy')
        #self.audio_file = os.path.join(self.root_path, 'audio_' + self.folds +'.npy')
  
        self.video = npy_loader(self.video_file)
        self.video_value = torch.ones(48, dtype=torch.long)
        #self.audio = npy_loader(self.audio_file)

        if self.folds == 'test':
            self.video = self.video[:10]


        # pre-processing the names
        self.char_dict = ['<pad>'] + [ _ for _ in string.ascii_uppercase ] + ['<eow>']
        self.max_seq = 13
        self.labels_char = self.labels_name
        self.labels_name = pad_words(self.char_dict, self.max_seq, self.labels_name)
        
        print('Loading {} part'.format(self.folds))

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        vid = self.augmentation(self.video[idx, :, :, :]).clamp(0,255)
        vid_face, vid_mouth = vid[:,:48,:], vid[:,48:,:]

        #vid_face = torch.stack([torch.sparse.FloatTensor(i.t(), self.video_value, torch.Size([256,256])).to_dense() for i in vid_face])
        #aud = self.audio[idx, :, :]

        vid_face = matrix_coordinate_transform(vid_face)
        label = self.labels_name[np.searchsorted( self.labels_interval, idx, side = 'left')]
        label_char = self.labels_char[np.searchsorted( self.labels_interval, idx, side = 'left')]

        return {"train_inputs":vid_face, "train_targets":vid_mouth}, label, label_char
        
    def augmentation(self, keypoints):
        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        if self.folds == 'train':
            d = keypoints_move + ones * randint
        else:
            d = keypoints_move + ones * 38
        return d.long()

def npy_loader(file):
    return torch.tensor(np.load(file))
