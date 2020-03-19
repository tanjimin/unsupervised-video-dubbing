import os
import cv2 
import torch
import math
import string

import numpy as np
from torch import nn

class CharacterEncoder( nn.Module ):
    """
    input: target word
    output: encoded chracter sequence
    """
    def __init__(self, max_seq = 13):
        super(CharacterEncoder, self).__init__()
        self.max_seq = max_seq
        self.embedding = nn.Sequential(
            nn.Embedding(28, 16), # 26 characters, 1 padding character, 1 eow. batch * max_seq * 16
        )
        self.convTrans = nn.Sequential(
            nn.ConvTranspose1d(self.max_seq, 29, 3, 1 ,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(29),
        )

        
        self.conv = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1), 
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
        )
    
    # def _pad_words(self, words):
    #     seq_tensors = [ [ self.char_dict.index(i) for i in word ] + [ len(self.char_dict) - 1 ] for word in words ]
    #     pre_pad_len = [ (self.max_seq - len(seq_tensor))//2 for seq_tensor in seq_tensors ]
    #     pst_pad_len = [ (self.max_seq - len(seq_tensor) + 1)//2 for seq_tensor in seq_tensors ]
    #     seq_tensors = [ [0] * pre_pad_len[i] + seq_tensor + [0] * pst_pad_len[i] for i,seq_tensor in seq_tensors ]
    #     seq_tensors = torch.tensor(seq_tensors).cuda()
    #     return seq_tensors # batch * max_seq


    def forward(self, seq_tensors):
        #seq_tensors = self._pad_words(words)
        seq_embedding = self.embedding(seq_tensors) # bach*13*16
        seq_embedding = self.convTrans(seq_embedding) # batch * 29 * 16
        seq_embedding = self.conv(seq_embedding.transpose(1,2)).transpose(1,2) #batch * 29 * 32
        return seq_embedding

class VAE(nn.Module):
    def __init__(self, mode, use_gpu ):
        super(VAE, self).__init__()
        self.mode = mode
        self.use_gpu = use_gpu
        self.K = 1
        self.inputDim = 1
        self.outputDim = self.inputDim
        self.inputHeight = 256
        self.convEncoder = nn.Sequential(
            nn.Conv2d(self.inputDim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.muEncoder    = nn.Linear(128 * (self.inputHeight // 4) * (self.inputHeight // 4), 128)
        self.sigmaEncoder = nn.Linear(128 * (self.inputHeight // 4) * (self.inputHeight // 4), 128)
        
        self.convDecoder1 = nn.Sequential(
            nn.Linear( 128, 128 * (self.inputHeight // 4) * (self.inputHeight // 4)),
            nn.BatchNorm1d( 128 * (self.inputHeight // 4) * (self.inputHeight // 4)),
            nn.LeakyReLU(0.2),
        )

        self.convDecoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, self.outputDim, 4, 2, 1),
            nn.Sigmoid(),
        )
        self._initialize_weights()
        
    def _sample(self, mu, sig):
        std = torch.exp(sig*0.5)
        noise = torch.randn(std.size())
        if self.use_gpu:
            noise = noise.cuda()
        return noise.add(mu) #noise.mul(std).add(mu)
    
    def forward(self, x):
        #inference
        x = self.convEncoder(x)
        x = x.view(-1, 128 * (self.inputHeight // 4) * (self.inputHeight // 4))
        mu = self.muEncoder(x)
        mu = mu.repeat(self.K,1,1).permute(1,0,2)
        sig = self.sigmaEncoder(x)
        sig = sig.repeat(self.K,1,1).permute(1,0,2)

        #sampling
        z = self._sample(mu,sig)
        #generate
        output = self.convDecoder1(z.view([-1,z.size()[-1]]))
        output = output.view(-1, 128, (self.inputHeight // 4), (self.inputHeight // 4))
        output = self.convDecoder2(output) #.unsqueeze(1)
        output = output.view([z.size()[0],z.size()[1],-1,self.inputHeight, self.inputHeight])
        return output, mu, sig
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CondRecurrentLG(VAE):
    def __init__(self, mode, use_gpu):
        super(CondRecurrentLG, self).__init__(mode, use_gpu)
        self.labelEncoder = nn.Sequential(
            nn.Embedding(500, 256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.LSTM = nn.Sequential(
            nn.LSTM(128 + 32, 128, 3, batch_first= True),
        )
        self.Linear =  nn.Sequential( 
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        self.Output = nn.Sequential(
            nn.Linear(256 * 256,128),
            nn.Tanh(),
            nn.Linear(128,40),
            nn.Sigmoid(),
        )

        self.seqLen = 29
        
    def forward(self, x, char_embd):
        # x: 29 * 2, 1 , 256, 256
        x = self.convEncoder(x) 
        x = x.view(-1, 128 * (self.inputHeight // 4) * (self.inputHeight // 4))
        mu = self.muEncoder(x)
        mu = mu.repeat(self.K,1,1).permute(1,0,2) # 29*2, 1, 128 
        sig = self.sigmaEncoder(x)
        sig = sig.repeat(self.K,1,1).permute(1,0,2)

        # sampling
        z = self._sample(mu,sig).view(-1, 29, 128)  # 2, 29, 128
        
        # concat to shape batch * 29 * (128 + 32)
        embeddings = torch.cat((z, char_embd), 2)
              
        # lstm
        hiddens, _ = self.LSTM(embeddings) # 2, 29, 128
        outputs = self.Linear(hiddens.contiguous().view(-1,128))
        
        # decode
        outputs = self.convDecoder1(outputs.view([-1,outputs.size()[-1]]))
        outputs = outputs.view(-1, 128, (self.inputHeight // 4), (self.inputHeight // 4))     
        outputs = self.convDecoder2(outputs)

        # turn to coordinate values
        outputs = outputs.view([z.size()[0],z.size()[1], 256 * 256 ])
        # 2, 29, 256 * 256
        outputs = 256 * self.Output(outputs) # 2 * 29 * 40
        return outputs, mu, sig